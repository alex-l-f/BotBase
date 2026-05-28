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
    BACKENDS,
)
from prompts import list_profiles
from prompts.topics import TOPICS, ROUTER_MODE
import argparse
import os
import json
import mimetypes
mimetypes.add_type('application/javascript', '.mjs')
mimetypes.add_type('application/javascript', '.js')

import sqlite3
from datetime import datetime
import threading

# Where the source files referenced by `source_path` live.
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
PROCESSED_ROOT = os.path.join(os.path.dirname(__file__), 'processed_resources')


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

app = Flask(__name__)
CORS(app)


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

    response_text, full_text, full_context = get_LM_response(
        data.get('fullContext', []), chat_id, model, profile=effective_profile
    )

    logger.log_event(chat_id, "send_response")

    return jsonify({
        "chat_id": chat_id,
        "profile": effective_profile,
        "mode": status.get("mode"),
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
    print(f"[server] LLM backend: {get_active_backend()}")
    app.run(host=args.host, port=args.port)
