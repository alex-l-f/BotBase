from flask import Flask, request, jsonify, send_from_directory, send_file, abort
from flask_cors import CORS
from agent import (
    get_LM_response,
    create_chat_session,
    get_messages,
    is_chat_complete,
    reset_complete,
    chat_status,
    set_backend,
    get_active_backend,
    memory_store,
    BACKENDS,
)
from prompts import list_profiles, ARCHITECTURES
from prompts.topics import TOPICS, ROUTER_MODE
import argparse
import base64
import hashlib
import os
import json
import mimetypes
import re
from xml.etree import ElementTree
mimetypes.add_type('application/javascript', '.mjs')
mimetypes.add_type('application/javascript', '.js')

import sqlite3
from datetime import datetime
import threading

import scenarios

# Where the source files referenced by `source_path` live.
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
PROCESSED_ROOT = os.path.join(os.path.dirname(__file__), 'processed_resources')

# SCORM/e-learning packages are dropped into the project root as directories
# containing an imsmanifest.xml (e.g. an Articulate Rise export).
SCORM_ROOT = os.path.dirname(os.path.abspath(__file__))


class EventLogger:
    def __init__(self, db_path="logs.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_logs (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    event_type TEXT,
                    data TEXT,
                    timestamp REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp
                ON session_logs(session_id, timestamp)
            """)
            conn.execute("PRAGMA journal_mode=WAL")

    def log_event(self, session_id, event_type, data=""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO session_logs (session_id, event_type, data, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, event_type, data, datetime.now().timestamp())
            )


logger = EventLogger()

# Which agent architecture new requests use unless the client asks for a
# specific one: 'multi' (coach + summarizer + memory) or 'single' (the
# original one-agent librarian, kept as the benchmark baseline).
DEFAULT_ARCH = os.getenv("BOTBASE_ARCH", "multi")

app = Flask(__name__)
CORS(app)

# --- Demo password gate ---
# Not real security: just enough to keep crawlers and passers-by from
# burning LLM tokens. The password check sets a long-lived cookie whose
# value is a digest of the password, so restarting the server doesn't
# log everyone out.
DEMO_PASSWORD = os.getenv("BOTBASE_PASSWORD", "AMIRADemo123")
AUTH_COOKIE = "botbase_auth"
AUTH_TOKEN = hashlib.sha256(DEMO_PASSWORD.encode()).hexdigest()


@app.before_request
def require_password():
    if request.method == 'OPTIONS' or request.path == '/login':
        return None
    if request.cookies.get(AUTH_COOKIE) == AUTH_TOKEN:
        return None
    if request.path == '/':
        return send_from_directory('static', 'login.html'), 401
    if request.path == '/memory':
        # Serve the (static, harmless) browser shell; its API calls will
        # 401 and the page shows a link back to the login page.
        return send_from_directory('static', 'memory.html'), 401
    return jsonify({"error": "Not authenticated"}), 401


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    if data.get('password') == DEMO_PASSWORD:
        resp = jsonify({"success": True})
        resp.set_cookie(
            AUTH_COOKIE, AUTH_TOKEN,
            max_age=30 * 24 * 3600, httponly=True, samesite='Lax',
        )
        return resp
    return jsonify({"success": False}), 401


@app.route('/api/start-chat', methods=['POST'])
def start_chat():
    chat_id = create_chat_session()
    logger.log_event(chat_id, "start_chat")
    return jsonify({"chat_id": chat_id})


@app.route('/api/get-messages/<chat_id>', methods=['GET'])
def get_chat_messages(chat_id):
    messages = get_messages(chat_id)
    is_complete = is_chat_complete(chat_id)
    return jsonify({
        "messages": messages,
        "is_complete": is_complete,
    })


@app.route('/api/reset-complete/<chat_id>', methods=['POST'])
def reset_chat_complete(chat_id):
    reset_complete(chat_id)
    return jsonify({"success": True})



@app.route('/api/prompt-chat', methods=['POST'])
def prompt_chat():
    """Chat with a custom system prompt."""
    data = request.json

    chat_id = data.get('chat_id')
    if not chat_id:
        chat_id = create_chat_session()

    logger.log_event(chat_id, "request_response")

    model = data.get('model')
    custom_prompt = data.get('prompt', None)

    response_text, full_text, full_context = get_LM_response(
        data.get('fullContext', []), chat_id, model, custom_prompt
    )

    logger.log_event(chat_id, "send_response")

    return jsonify({
        "chat_id": chat_id,
        "message": {"content": response_text},
        "messages": {"content": full_text, "full_context": full_context}
    })


@app.route('/api/profiles', methods=['GET'])
def api_profiles():
    """List all available prompt/toolset profiles."""
    return jsonify({"profiles": list_profiles()})


@app.route('/api/chat-profile', methods=['POST'])
def chat_profile():
    """Chat using a named profile (prompt + toolset combination).

    For topic-aware profiles, the *current* mode is held server-side in
    chat_status[chat_id]['mode'] so it survives across turns even though
    each request is stateless from the client's perspective. The first
    call seeds the mode from the request; subsequent calls use whatever
    switch_mode last set, ignoring the client-supplied profile."""
    data = request.json
    chat_id = data.get('chat_id')
    if not chat_id:
        chat_id = create_chat_session()

    logger.log_event(chat_id, "request_response")

    model = data.get('model')
    requested_profile = data.get('profile', "default")

    arch = data.get('arch') or DEFAULT_ARCH
    if arch not in ARCHITECTURES:
        arch = DEFAULT_ARCH

    # Resolve effective profile for this turn.
    status = chat_status.setdefault(chat_id, {"is_complete": False})
    stored_mode = status.get("mode")
    if stored_mode in TOPICS:
        effective_profile = stored_mode
    elif requested_profile in TOPICS:
        # First topic-aware call for this chat — seed the stored mode.
        status["mode"] = requested_profile
        effective_profile = requested_profile
    else:
        # Legacy non-topic profile (e.g. 'default'); don't track mode.
        effective_profile = requested_profile

    # A finished turn leaves is_complete=True behind; clear it so the
    # frontend's polling loop doesn't stop on its first fetch of the new
    # turn (which silently dropped queued file/mode events).
    status["is_complete"] = False

    response_text, full_text, full_context = get_LM_response(
        data.get('fullContext', []), chat_id, model, profile=effective_profile,
        arch=arch
    )

    logger.log_event(chat_id, "send_response")

    return jsonify({
        "chat_id": chat_id,
        "profile": effective_profile,
        "mode": status.get("mode"),
        "arch": arch,
        "message": {"content": response_text},
        "messages": {"content": full_text, "full_context": full_context}
    })


@app.route('/api/get-mode/<chat_id>', methods=['GET'])
def get_mode(chat_id):
    """Return the chatbot's current topic mode for this chat."""
    status = chat_status.get(chat_id) or {}
    mode = status.get("mode")
    label = TOPICS.get(mode, {}).get("label") if mode else None
    return jsonify({
        "mode": mode,
        "label": label,
    })


@app.route('/api/topics', methods=['GET'])
def get_topics():
    """List available topic modes (for any UI that wants them)."""
    return jsonify({
        "router": ROUTER_MODE,
        "arch_default": DEFAULT_ARCH,
        "architectures": list(ARCHITECTURES),
        "topics": [
            {
                "key": k,
                "label": v["label"],
                "description": v["short_description"],
                "provider": v["provider"],
            }
            for k, v in TOPICS.items()
        ],
    })


@app.route('/api/file/<provider>/<int:resource_id>', methods=['GET'])
def serve_resource_file(provider, resource_id):
    """Stream a topic resource file by (provider, integer resource id).

    The portal_url stored in each topic's database.db points at this route,
    so search_resources output already carries working URLs."""
    db_path = os.path.join(PROCESSED_ROOT, provider, 'database.db')
    if not os.path.exists(db_path):
        abort(404, description=f"Unknown provider: {provider}")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT source_file, source_path, source_type "
            "FROM resources WHERE id = ?",
            (resource_id,),
        ).fetchone()
        conn.close()
    except sqlite3.Error as exc:
        abort(500, description=f"Database error: {exc}")

    if row is None:
        abort(404, description=f"Resource {resource_id} not found in {provider}")

    source_path = row["source_path"] or ""
    if not source_path:
        abort(404, description="Resource has no source_path")

    # Resolve under DATA_ROOT and ensure no path-traversal escapes happen.
    abs_path = os.path.abspath(os.path.join(DATA_ROOT, source_path))
    data_root_abs = os.path.abspath(DATA_ROOT)
    if not abs_path.startswith(data_root_abs + os.sep) and abs_path != data_root_abs:
        abort(400, description="Invalid source path")
    if not os.path.isfile(abs_path):
        abort(404, description=f"File missing on disk: {source_path}")

    mime, _ = mimetypes.guess_type(abs_path)
    return send_file(
        abs_path,
        mimetype=mime,
        as_attachment=False,
        download_name=row["source_file"] or os.path.basename(abs_path),
        conditional=True,
    )


def _find_scorm_packages():
    """Directories in the project root that look like SCORM packages."""
    packages = []
    for name in sorted(os.listdir(SCORM_ROOT)):
        pkg_dir = os.path.join(SCORM_ROOT, name)
        if os.path.isdir(pkg_dir) and os.path.isfile(os.path.join(pkg_dir, 'imsmanifest.xml')):
            packages.append(name)
    return packages


def _manifest_title(pkg_dir):
    """Course title from imsmanifest.xml (namespace-agnostic)."""
    try:
        tree = ElementTree.parse(os.path.join(pkg_dir, 'imsmanifest.xml'))
        for el in tree.iter():
            if el.tag.split('}')[-1] == 'title' and el.text and el.text.strip():
                return el.text.strip()
    except (ElementTree.ParseError, OSError):
        pass
    return None


def _parse_rise_course(pkg_dir):
    """Decode the course JSON an Articulate Rise export embeds in
    scormcontent/index.html as a base64 blob passed to deserialize()."""
    index_path = os.path.join(pkg_dir, 'scormcontent', 'index.html')
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, encoding='utf-8') as f:
            html = f.read()
        m = re.search(r'deserialize\("([^"]+)"\)', html)
        if not m:
            return None
        return json.loads(base64.b64decode(m.group(1))).get('course')
    except (OSError, ValueError):
        return None


_scorm_index_cache = None


@app.route('/api/scorm/index', methods=['GET'])
def scorm_index():
    """Index of all SCORM packages: sections and deep-linkable lessons.

    URLs are relative to the app mount point so the frontend can prefix
    them with its API base (works behind a path-prefixed reverse proxy)."""
    global _scorm_index_cache
    if _scorm_index_cache is None:
        modules = []
        for pkg in _find_scorm_packages():
            pkg_dir = os.path.join(SCORM_ROOT, pkg)
            content_url = f"scorm/{pkg}/scormcontent/index.html"
            title = _manifest_title(pkg_dir) or pkg
            sections = []
            course = _parse_rise_course(pkg_dir)
            if course:
                title = course.get('title') or title
                current = None
                for lesson in course.get('lessons', []):
                    if lesson.get('type') == 'section':
                        current = {"title": lesson.get('title', ''), "lessons": []}
                        sections.append(current)
                        continue
                    if current is None:
                        current = {"title": "", "lessons": []}
                        sections.append(current)
                    current["lessons"].append({
                        "id": lesson.get('id'),
                        "title": lesson.get('title', ''),
                        "url": f"{content_url}#/lessons/{lesson.get('id')}",
                    })
            modules.append({
                "package": pkg,
                "title": title,
                "url": content_url,
                "sections": sections,
            })
        _scorm_index_cache = modules
    return jsonify({"modules": _scorm_index_cache})


@app.route('/scorm/<package>/<path:filename>')
def serve_scorm(package, filename):
    """Serve files out of a SCORM package directory."""
    pkg_dir = os.path.join(SCORM_ROOT, package)
    if not os.path.isfile(os.path.join(pkg_dir, 'imsmanifest.xml')):
        abort(404, description=f"Unknown SCORM package: {package}")
    return send_from_directory(pkg_dir, filename)


# ------------------------------------------------------------ memory browser
# Read/edit surface over the episodic memory store, for demos ("show what
# the bot remembers") and for staging demo state. All writes go through
# MemoryStore's admin methods so it stays the single writer.

@app.route('/memory')
def serve_memory_browser():
    return send_from_directory('static', 'memory.html')


@app.route('/api/memory/stats', methods=['GET'])
def memory_stats():
    return jsonify(memory_store.stats())


@app.route('/api/memory/snapshot', methods=['GET'])
def memory_snapshot():
    """The MEMORY SNAPSHOT block exactly as it would be injected into the
    coach's system prompt right now (optionally excluding one chat)."""
    block = memory_store.profile_block(
        exclude_chat=request.args.get('exclude_chat'))
    return jsonify({"block": block})


@app.route('/api/memory/episodes', methods=['GET'])
def memory_episodes_list():
    try:
        limit = max(1, min(200, int(request.args.get('limit', 30))))
        offset = max(0, int(request.args.get('offset', 0)))
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400
    result = memory_store.list_episodes(
        query=request.args.get('query'),
        mode=request.args.get('mode'),
        kind=request.args.get('kind'),
        chat=request.args.get('chat'),
        limit=limit, offset=offset,
    )
    return jsonify(result)


@app.route('/api/memory/episodes', methods=['POST'])
def memory_episodes_create():
    data = request.get_json(silent=True) or {}
    try:
        eid = memory_store.add_episode(
            chat_id=(data.get('chat_id') or 'manual').strip(),
            mode=(data.get('mode') or '').strip() or None,
            kind=(data.get('kind') or '').strip(),
            content=data.get('content') or '',
        )
        return jsonify({"id": eid})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/api/memory/episodes/<int:episode_id>', methods=['PUT'])
def memory_episodes_update(episode_id):
    data = request.get_json(silent=True) or {}
    try:
        ok = memory_store.update_episode(
            episode_id,
            content=data.get('content'),
            mode=(data['mode'].strip() or None) if 'mode' in data else None,
            kind=data.get('kind'),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not ok:
        return jsonify({"error": "Episode not found"}), 404
    return jsonify({"success": True})


@app.route('/api/memory/episodes/<int:episode_id>', methods=['DELETE'])
def memory_episodes_delete(episode_id):
    if not memory_store.delete_episode(episode_id):
        return jsonify({"error": "Episode not found"}), 404
    return jsonify({"success": True})


# Scenario snapshots: save the current episodic log under a name, reload it
# later (optionally re-dated) to replay the same demo state. Stored as JSON
# files under scenarios/ — see scenarios.py.

@app.route('/api/memory/scenarios', methods=['GET'])
def memory_scenarios_list():
    return jsonify({"scenarios": scenarios.list_scenarios()})


@app.route('/api/memory/scenarios', methods=['POST'])
def memory_scenarios_save():
    data = request.get_json(silent=True) or {}
    try:
        slug = scenarios.save_scenario(
            data.get('name') or '',
            data.get('description') or '',
            memory_store.export_episodes(),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"slug": slug})


@app.route('/api/memory/scenarios/<slug>/load', methods=['POST'])
def memory_scenarios_load(slug):
    try:
        scenario = scenarios.get_scenario(slug)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if scenario is None:
        return jsonify({"error": "Scenario not found"}), 404

    data = request.get_json(silent=True) or {}
    episodes = scenario.get("episodes") or []
    days = data.get('retime_newest_days_ago')
    if days is not None:
        try:
            episodes = scenarios.retime_episodes(episodes, float(days))
        except (TypeError, ValueError):
            return jsonify({"error": "retime_newest_days_ago must be a number"}), 400
    try:
        loaded = memory_store.replace_episodes(episodes)
    except ValueError as exc:
        return jsonify({"error": f"Scenario file invalid: {exc}"}), 400
    return jsonify({"loaded": loaded, "name": scenario.get("name") or slug})


@app.route('/api/memory/scenarios/<slug>', methods=['DELETE'])
def memory_scenarios_delete(slug):
    try:
        ok = scenarios.delete_scenario(slug)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not ok:
        return jsonify({"error": "Scenario not found"}), 404
    return jsonify({"success": True})


@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/api/log-event', methods=['POST'])
def log_event_endpoint():
    data = request.json
    chat_id = data.get('chat_id', 'unknown')
    event_type = data.get('event_type', '')
    event_data = data.get('data', '')
    logger.log_event(chat_id, event_type, str(event_data))
    return jsonify({'success': True})


def _parse_args():
    parser = argparse.ArgumentParser(description="BotBase chat server.")
    parser.add_argument(
        "--backend", "-b",
        choices=sorted(BACKENDS.keys()),
        default=os.getenv("BOTBASE_BACKEND", "openrouter"),
        help=(
            "Which LLM backend to use. 'openrouter' calls OpenRouter's API "
            "(needs OPENROUTER_API_KEY in .env); 'llama_cpp' targets a local "
            "llama.cpp server. Defaults to $BOTBASE_BACKEND, then 'openrouter'."
        ),
    )
    parser.add_argument(
        "--arch", "-a",
        choices=list(ARCHITECTURES),
        default=os.getenv("BOTBASE_ARCH", "multi"),
        help=(
            "Default agent architecture: 'multi' runs the coach + summarizer "
            "+ memory system; 'single' runs the original one-agent librarian "
            "(the benchmark baseline). Clients can override per request. "
            "Defaults to $BOTBASE_ARCH, then 'multi'."
        ),
    )
    parser.add_argument(
        "--host", default=os.getenv("BOTBASE_HOST", "0.0.0.0"),
        help="Host to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("BOTBASE_PORT", "5551")),
        help="Port to bind (default: 5551).",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    set_backend(args.backend)
    DEFAULT_ARCH = args.arch
    print(f"[server] LLM backend: {get_active_backend()}")
    print(f"[server] agent architecture: {DEFAULT_ARCH}")
    app.run(host=args.host, port=args.port)
