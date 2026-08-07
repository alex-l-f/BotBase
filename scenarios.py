"""
Named memory scenarios for demos.

A scenario is a snapshot of the episodic memory log saved as a JSON file
under scenarios/. Files are portable (sync them to another machine, commit
them, hand-edit them) and human-readable:

    {"name": ..., "description": ..., "saved_at": ..., "episodes": [...]}

Loading a scenario replaces the live episodic log via
MemoryStore.replace_episodes (still the single writer). The retime helper
shifts every timestamp by the same delta so the *newest* entry lands N days
before now — relative spacing between entries is preserved, which keeps
"user returning the day after a session" true no matter when the demo runs.
"""

import json
import os
import re
from datetime import datetime, timedelta

SCENARIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")

_SLUG_RE = re.compile(r"^[a-z0-9_-]+$")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().lower()).strip("-")
    if not slug:
        raise ValueError("Scenario name must contain letters or digits")
    return slug[:60]


def _path(slug: str) -> str:
    if not _SLUG_RE.match(slug):
        raise ValueError(f"Invalid scenario id {slug!r}")
    return os.path.join(SCENARIO_DIR, f"{slug}.json")


def list_scenarios() -> list[dict]:
    if not os.path.isdir(SCENARIO_DIR):
        return []
    out = []
    for fname in sorted(os.listdir(SCENARIO_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SCENARIO_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "slug": fname[:-5],
            "name": data.get("name") or fname[:-5],
            "description": data.get("description") or "",
            "saved_at": data.get("saved_at") or "",
            "episodes": len(data.get("episodes") or []),
        })
    out.sort(key=lambda s: s["saved_at"], reverse=True)
    return out


def save_scenario(name: str, description: str, episodes: list[dict]) -> str:
    slug = _slugify(name)
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    with open(_path(slug), "w", encoding="utf-8") as f:
        json.dump({
            "name": name.strip(),
            "description": (description or "").strip(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "episodes": episodes,
        }, f, indent=2, ensure_ascii=False)
    return slug


def get_scenario(slug: str) -> dict | None:
    try:
        with open(_path(slug), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def delete_scenario(slug: str) -> bool:
    try:
        os.remove(_path(slug))
        return True
    except OSError:
        return False


def retime_episodes(episodes: list[dict], newest_days_ago: float) -> list[dict]:
    """Shift all parseable timestamps by one shared delta so the newest one
    becomes now - newest_days_ago. Unparseable timestamps pass through."""
    parsed = {}
    for i, ep in enumerate(episodes):
        try:
            parsed[i] = datetime.fromisoformat(ep.get("ts") or "")
        except (ValueError, TypeError):
            pass
    if not parsed:
        return episodes
    newest = max(parsed.values())
    delta = (datetime.now() - timedelta(days=newest_days_ago)) - newest
    out = []
    for i, ep in enumerate(episodes):
        ep = dict(ep)
        if i in parsed:
            ep["ts"] = (parsed[i] + delta).isoformat(timespec="seconds")
        out.append(ep)
    return out
