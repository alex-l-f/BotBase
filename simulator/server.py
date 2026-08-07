"""Standalone bot-simulator server.

Serves the review UI and the simulator API. Talks to a running BotBase
instance purely over HTTP — no code dependency on the main project.

Run:  python server.py  (default port 5561)
"""

import argparse
import hashlib
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import db
import runner
from bot_client import BotClient, BotClientError

app = Flask(__name__)
CORS(app)

db.init_db()

# --- Shared password gate (same style as BotBase's demo gate) -----------
SIM_PASSWORD = os.getenv("SIM_PASSWORD", "SimReview123")
AUTH_COOKIE = "sim_auth"
AUTH_TOKEN = hashlib.sha256(("sim-salt::" + SIM_PASSWORD).encode()).hexdigest()

# Defaults offered to the UI for new runs.
DEFAULT_BOT_URL = os.getenv("SIM_BOT_URL", "http://localhost:5551")
DEFAULT_BOT_PASSWORD = os.getenv("SIM_BOT_PASSWORD", "AMIRADemo123")


@app.before_request
def require_password():
    if request.method == "OPTIONS" or request.path == "/login":
        return None
    if request.cookies.get(AUTH_COOKIE) == AUTH_TOKEN:
        return None
    if request.path == "/":
        return send_from_directory("static", "login.html"), 401
    return jsonify({"error": "Not authenticated"}), 401


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    if data.get("password") == SIM_PASSWORD:
        resp = jsonify({"success": True})
        resp.set_cookie(AUTH_COOKIE, AUTH_TOKEN,
                        max_age=30 * 24 * 3600, httponly=True, samesite="Lax")
        return resp
    return jsonify({"success": False}), 401


@app.route("/")
def serve_frontend():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "default_bot_url": DEFAULT_BOT_URL,
        "model_suggestions": [
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-haiku-4.5",
            "openai/gpt-5.1",
            "openai/gpt-5-mini",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat-v3.1",
        ],
    })


# ---------------------------------------------------------------- personas

@app.route("/api/personas", methods=["GET"])
def personas_list():
    return jsonify({"personas": db.list_personas()})


@app.route("/api/personas", methods=["POST"])
def personas_create():
    data = request.get_json(force=True)
    err = _validate_persona(data)
    if err:
        return jsonify({"error": err}), 400
    pid = db.create_persona(data)
    return jsonify({"id": pid})


@app.route("/api/personas/<int:pid>", methods=["PUT"])
def personas_update(pid):
    data = request.get_json(force=True)
    err = _validate_persona(data)
    if err:
        return jsonify({"error": err}), 400
    if not db.get_persona(pid):
        return jsonify({"error": "Not found"}), 404
    db.update_persona(pid, data)
    return jsonify({"success": True})


@app.route("/api/personas/<int:pid>", methods=["DELETE"])
def personas_delete(pid):
    db.archive_persona(pid)
    return jsonify({"success": True})


def _validate_persona(data):
    for f in ("name", "profile_text", "model"):
        if not (data.get(f) or "").strip():
            return f"Missing required field: {f}"
    return None


# ----------------------------------------------------------------- rubrics

@app.route("/api/rubrics", methods=["GET"])
def rubrics_list():
    return jsonify({"rubrics": db.list_rubrics()})


@app.route("/api/rubrics", methods=["POST"])
def rubrics_create():
    data = request.get_json(force=True)
    err = _validate_rubric(data)
    if err:
        return jsonify({"error": err}), 400
    rid = db.create_rubric(data)
    return jsonify({"id": rid})


@app.route("/api/rubrics/<int:rid>", methods=["PUT"])
def rubrics_update(rid):
    data = request.get_json(force=True)
    err = _validate_rubric(data)
    if err:
        return jsonify({"error": err}), 400
    if not db.get_rubric(rid):
        return jsonify({"error": "Not found"}), 404
    db.update_rubric(rid, data)
    return jsonify({"success": True})


@app.route("/api/rubrics/<int:rid>", methods=["DELETE"])
def rubrics_delete(rid):
    db.archive_rubric(rid)
    return jsonify({"success": True})


def _validate_rubric(data):
    if not (data.get("name") or "").strip():
        return "Missing required field: name"
    if not (data.get("judge_model") or "").strip():
        return "Missing required field: judge_model"
    criteria = data.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        return "Rubric needs at least one criterion"
    seen = set()
    for c in criteria:
        for f in ("key", "title", "description"):
            if not (c.get(f) or "").strip():
                return f"Criterion missing field: {f}"
        if c["key"] in seen:
            return f"Duplicate criterion key: {c['key']}"
        seen.add(c["key"])
        try:
            lo, hi = float(c.get("min", 1)), float(c.get("max", 5))
            float(c.get("weight", 1))
        except (TypeError, ValueError):
            return f"Criterion {c['key']}: min/max/weight must be numbers"
        if hi <= lo:
            return f"Criterion {c['key']}: max must be greater than min"
    return None


# -------------------------------------------------------------------- runs

@app.route("/api/runs", methods=["GET"])
def runs_list():
    runs = db.list_runs()
    for r in runs:
        if r["status"] in ("running", "evaluating", "pending") and not runner.is_run_active(r["id"]):
            # Server restarted mid-run; mark it dead so the UI doesn't spin forever.
            db.update_run(r["id"], status="failed", error="Run interrupted (server restart)")
            r["status"] = "failed"
    return jsonify({"runs": runs})


@app.route("/api/runs", methods=["POST"])
def runs_create():
    data = request.get_json(force=True)

    persona = db.get_persona(data.get("persona_id") or -1)
    if not persona:
        return jsonify({"error": "Persona not found"}), 400

    rubric = None
    if data.get("rubric_id"):
        rubric = db.get_rubric(data["rubric_id"])
        if not rubric:
            return jsonify({"error": "Rubric not found"}), 400

    bot_url = (data.get("bot_url") or DEFAULT_BOT_URL).strip()
    bot_profile = (data.get("bot_profile") or "default").strip()
    bot_password = data.get("bot_password", DEFAULT_BOT_PASSWORD)
    # '' means "use the bot server's default architecture"; anything else is
    # sent through verbatim (the bot validates it against its own list).
    bot_arch = (data.get("bot_arch") or "").strip()
    max_turns = max(1, min(50, int(data.get("max_turns", 10))))

    run_id = db.create_run({
        "name": data.get("name", ""),
        "created_by": data.get("created_by", ""),
        "persona_id": persona["id"],
        "persona_snapshot": persona,
        "rubric_id": rubric["id"] if rubric else None,
        "rubric_snapshot": rubric,
        "bot_url": bot_url,
        "bot_profile": bot_profile,
        "bot_arch": bot_arch,
        "bot_backend_note": data.get("bot_backend_note", ""),
        "max_turns": max_turns,
    })

    runner.start_run(run_id, {
        "persona": persona,
        "rubric": rubric,
        "bot_url": bot_url,
        "bot_profile": bot_profile,
        "bot_arch": bot_arch,
        "bot_password": bot_password,
        "max_turns": max_turns,
    })
    return jsonify({"id": run_id})


@app.route("/api/runs/<int:run_id>", methods=["GET"])
def runs_get(run_id):
    run = db.get_run(run_id)
    if not run:
        return jsonify({"error": "Not found"}), 404
    if run["status"] in ("running", "evaluating", "pending") and not runner.is_run_active(run_id):
        db.update_run(run_id, status="failed", error="Run interrupted (server restart)")
        run["status"] = "failed"
    return jsonify(run)


@app.route("/api/runs/<int:run_id>", methods=["DELETE"])
def runs_delete(run_id):
    if runner.is_run_active(run_id):
        return jsonify({"error": "Run is still in progress"}), 409
    db.delete_run(run_id)
    return jsonify({"success": True})


@app.route("/api/runs/<int:run_id>/evaluate", methods=["POST"])
def runs_evaluate(run_id):
    """(Re-)evaluate an existing run, optionally with a different rubric/judge."""
    run = db.get_run(run_id)
    if not run:
        return jsonify({"error": "Not found"}), 404
    if not run.get("transcript"):
        return jsonify({"error": "Run has no transcript"}), 400

    data = request.get_json(silent=True) or {}
    rubric = db.get_rubric(data["rubric_id"]) if data.get("rubric_id") else run.get("rubric_snapshot")
    if not rubric:
        return jsonify({"error": "No rubric available — pass rubric_id"}), 400

    try:
        eval_id = runner.evaluate_run(
            run_id, run["transcript"], rubric, judge_model=data.get("judge_model"))
        return jsonify({"id": eval_id})
    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {e}"}), 502


# ----------------------------------------------------------------- reviews

@app.route("/api/runs/<int:run_id>/reviews", methods=["POST"])
def reviews_create(run_id):
    if not db.get_run(run_id):
        return jsonify({"error": "Run not found"}), 404
    data = request.get_json(force=True)
    reviewer = (data.get("reviewer") or "").strip()
    if not reviewer:
        return jsonify({"error": "Reviewer name is required"}), 400
    rating = data.get("rating")
    if rating is not None:
        rating = max(1, min(5, int(rating)))
    rid = db.create_review(run_id, reviewer, rating, (data.get("comment") or "").strip())
    return jsonify({"id": rid})


@app.route("/api/reviews/<int:review_id>", methods=["DELETE"])
def reviews_delete(review_id):
    db.delete_review(review_id)
    return jsonify({"success": True})


# ---------------------------------------------------------- bot proxying

@app.route("/api/bot/profiles", methods=["POST"])
def bot_profiles():
    """Fetch the profile + architecture lists from a BotBase server
    (handles its login)."""
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or DEFAULT_BOT_URL).strip()
    password = data.get("password", DEFAULT_BOT_PASSWORD)
    try:
        client = BotClient(url, password=password)
        profiles = client.list_profiles()
        try:
            arch_info = client.get_architectures()
        except BotClientError:
            # Pre-multi-agent server; profiles alone are still useful.
            arch_info = {"architectures": [], "default": None}
        return jsonify({
            "profiles": profiles,
            "architectures": arch_info["architectures"],
            "arch_default": arch_info["default"],
        })
    except BotClientError as e:
        return jsonify({"error": str(e)}), 502


def _parse_args():
    p = argparse.ArgumentParser(description="BotBase conversation simulator")
    p.add_argument("--port", type=int, default=5561)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"[simulator] http://localhost:{args.port}  (password: set SIM_PASSWORD to change)")
    app.run(host=args.host, port=args.port, threaded=True)
