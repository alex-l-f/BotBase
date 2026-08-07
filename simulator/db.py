"""SQLite storage for the bot simulator.

One connection per call (WAL mode) so the Flask threads and the
background run threads can all touch the database safely.
"""

import json
import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "simulator.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    short_desc TEXT DEFAULT '',
    profile_text TEXT NOT NULL,
    model TEXT NOT NULL,
    temperature REAL DEFAULT 0.8,
    created_at REAL,
    updated_at REAL,
    archived INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rubrics (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    criteria TEXT NOT NULL,          -- JSON: [{key,title,description,min,max,weight}]
    judge_model TEXT NOT NULL,
    created_at REAL,
    updated_at REAL,
    archived INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    name TEXT DEFAULT '',
    created_by TEXT DEFAULT '',
    persona_id INTEGER,
    persona_snapshot TEXT,           -- JSON copy of the persona at run time
    rubric_id INTEGER,
    rubric_snapshot TEXT,            -- JSON copy of the rubric at run time
    bot_url TEXT,
    bot_profile TEXT,
    bot_arch TEXT DEFAULT '',        -- agent architecture ('single'/'multi'; '' = server default)
    bot_backend_note TEXT DEFAULT '',
    max_turns INTEGER DEFAULT 10,
    status TEXT DEFAULT 'pending',   -- pending | running | evaluating | completed | failed
    error TEXT DEFAULT '',
    transcript TEXT DEFAULT '[]',    -- JSON: [{role: 'user'|'bot', content, ts}]
    full_context TEXT DEFAULT '[]',  -- raw BotBase full_context (incl. tool calls)
    bot_chat_id TEXT DEFAULT '',
    created_at REAL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    rubric_name TEXT DEFAULT '',
    rubric_snapshot TEXT,
    judge_model TEXT,
    scores TEXT,                     -- JSON: [{key,title,score,max,weight,justification}]
    overall_comment TEXT DEFAULT '',
    weighted_score REAL,             -- 0-100 normalized
    raw_response TEXT DEFAULT '',
    status TEXT DEFAULT 'completed', -- completed | failed
    error TEXT DEFAULT '',
    created_at REAL
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    reviewer TEXT NOT NULL,
    rating INTEGER,                  -- optional 1-5
    comment TEXT DEFAULT '',
    created_at REAL
);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript(SCHEMA)
        # Migration for databases created before the multi-agent split:
        # CREATE TABLE IF NOT EXISTS won't add new columns to old tables.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
        if "bot_arch" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN bot_arch TEXT DEFAULT ''")


def _row_to_dict(row, json_fields=()):
    d = dict(row)
    for f in json_fields:
        if f in d and isinstance(d[f], str):
            try:
                d[f] = json.loads(d[f])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


# ---------------------------------------------------------------- personas

def list_personas(include_archived=False):
    q = "SELECT * FROM personas"
    if not include_archived:
        q += " WHERE archived=0"
    q += " ORDER BY updated_at DESC"
    with _connect() as conn:
        return [dict(r) for r in conn.execute(q)]


def get_persona(pid):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM personas WHERE id=?", (pid,)).fetchone()
        return dict(row) if row else None


def create_persona(data):
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO personas (name, short_desc, profile_text, model, temperature, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (data["name"], data.get("short_desc", ""), data["profile_text"],
             data["model"], data.get("temperature", 0.8), now, now),
        )
        return cur.lastrowid


def update_persona(pid, data):
    with _connect() as conn:
        conn.execute(
            "UPDATE personas SET name=?, short_desc=?, profile_text=?, model=?, temperature=?, updated_at=? WHERE id=?",
            (data["name"], data.get("short_desc", ""), data["profile_text"],
             data["model"], data.get("temperature", 0.8), time.time(), pid),
        )


def archive_persona(pid):
    with _connect() as conn:
        conn.execute("UPDATE personas SET archived=1, updated_at=? WHERE id=?", (time.time(), pid))


# ----------------------------------------------------------------- rubrics

def list_rubrics(include_archived=False):
    q = "SELECT * FROM rubrics"
    if not include_archived:
        q += " WHERE archived=0"
    q += " ORDER BY updated_at DESC"
    with _connect() as conn:
        return [_row_to_dict(r, ("criteria",)) for r in conn.execute(q)]


def get_rubric(rid):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM rubrics WHERE id=?", (rid,)).fetchone()
        return _row_to_dict(row, ("criteria",)) if row else None


def create_rubric(data):
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO rubrics (name, description, criteria, judge_model, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (data["name"], data.get("description", ""), json.dumps(data["criteria"]),
             data["judge_model"], now, now),
        )
        return cur.lastrowid


def update_rubric(rid, data):
    with _connect() as conn:
        conn.execute(
            "UPDATE rubrics SET name=?, description=?, criteria=?, judge_model=?, updated_at=? WHERE id=?",
            (data["name"], data.get("description", ""), json.dumps(data["criteria"]),
             data["judge_model"], time.time(), rid),
        )


def archive_rubric(rid):
    with _connect() as conn:
        conn.execute("UPDATE rubrics SET archived=1, updated_at=? WHERE id=?", (time.time(), rid))


# -------------------------------------------------------------------- runs

def create_run(data):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (name, created_by, persona_id, persona_snapshot, rubric_id, rubric_snapshot, "
            "bot_url, bot_profile, bot_arch, bot_backend_note, max_turns, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending',?)",
            (data.get("name", ""), data.get("created_by", ""),
             data["persona_id"], json.dumps(data["persona_snapshot"]),
             data.get("rubric_id"), json.dumps(data.get("rubric_snapshot")),
             data["bot_url"], data["bot_profile"], data.get("bot_arch", ""),
             data.get("bot_backend_note", ""),
             data.get("max_turns", 10), time.time()),
        )
        return cur.lastrowid


def list_runs(limit=200):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT r.id, r.name, r.created_by, r.persona_id, r.rubric_id, r.bot_url, r.bot_profile, "
            "r.bot_arch, r.bot_backend_note, r.max_turns, r.status, r.error, r.created_at, r.completed_at, "
            "json_extract(r.persona_snapshot, '$.name') AS persona_name, "
            "json_extract(r.rubric_snapshot, '$.name') AS rubric_name, "
            "(SELECT COUNT(*) FROM json_each(r.transcript)) AS message_count, "
            "(SELECT weighted_score FROM evaluations e WHERE e.run_id = r.id AND e.status='completed' "
            " ORDER BY e.created_at DESC LIMIT 1) AS latest_score, "
            "(SELECT COUNT(*) FROM reviews v WHERE v.run_id = r.id) AS review_count "
            "FROM runs r ORDER BY r.created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]


def get_run(run_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        run = _row_to_dict(row, ("persona_snapshot", "rubric_snapshot", "transcript", "full_context"))
        run["evaluations"] = [
            _row_to_dict(r, ("scores", "rubric_snapshot"))
            for r in conn.execute(
                "SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC", (run_id,))
        ]
        run["reviews"] = [
            dict(r) for r in conn.execute(
                "SELECT * FROM reviews WHERE run_id=? ORDER BY created_at ASC", (run_id,))
        ]
        return run


def update_run(run_id, **fields):
    json_fields = {"transcript", "full_context", "persona_snapshot", "rubric_snapshot"}
    sets, vals = [], []
    for k, v in fields.items():
        sets.append(f"{k}=?")
        vals.append(json.dumps(v) if k in json_fields else v)
    vals.append(run_id)
    with _connect() as conn:
        conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", vals)


def delete_run(run_id):
    with _connect() as conn:
        conn.execute("DELETE FROM evaluations WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM reviews WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM runs WHERE id=?", (run_id,))


# ------------------------------------------------------------- evaluations

def create_evaluation(run_id, rubric_snapshot, judge_model, scores, overall_comment,
                      weighted_score, raw_response, status="completed", error=""):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO evaluations (run_id, rubric_name, rubric_snapshot, judge_model, scores, "
            "overall_comment, weighted_score, raw_response, status, error, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, (rubric_snapshot or {}).get("name", ""), json.dumps(rubric_snapshot),
             judge_model, json.dumps(scores), overall_comment, weighted_score,
             raw_response, status, error, time.time()),
        )
        return cur.lastrowid


# ----------------------------------------------------------------- reviews

def create_review(run_id, reviewer, rating, comment):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO reviews (run_id, reviewer, rating, comment, created_at) VALUES (?,?,?,?,?)",
            (run_id, reviewer, rating, comment, time.time()),
        )
        return cur.lastrowid


def delete_review(review_id):
    with _connect() as conn:
        conn.execute("DELETE FROM reviews WHERE id=?", (review_id,))
