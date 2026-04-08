from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from agent import get_LM_response, create_chat_session, get_messages, is_chat_complete, reset_complete
from prompts import list_profiles
import os
import json
import mimetypes
mimetypes.add_type('application/javascript', '.mjs')
mimetypes.add_type('application/javascript', '.js')

import sqlite3
from datetime import datetime
import threading


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
    """Chat using a named profile (prompt + toolset combination)."""
    data = request.json
    chat_id = data.get('chat_id')
    if not chat_id:
        chat_id = create_chat_session()

    logger.log_event(chat_id, "request_response")

    model = data.get('model')
    profile = data.get('profile', "default")

    response_text, full_text, full_context = get_LM_response(
        data.get('fullContext', []), chat_id, model, profile=profile
    )

    logger.log_event(chat_id, "send_response")

    return jsonify({
        "chat_id": chat_id,
        "profile": profile,
        "message": {"content": response_text},
        "messages": {"content": full_text, "full_context": full_context}
    })


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


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5551)
