"""
Template registry for the privacy-preserving memory system.

Design rule: nothing free-form is ever stored. A memory is a template key
plus slot values, and every slot is one of:
  - an enum value from the fixed vocabularies below,
  - a reference to a library resource (provider key + integer id, with the
    library title snapshotted for display), or
  - a topic key from prompts/topics.py.

The memory agent fills templates via the record_memories tool (structured
generation — the tool schema only admits these enums), the browser's add
form is driven by the same registry via /api/memory/templates, and the
MemoryStore re-validates on every write. Rendering back to human-readable
text is deterministic, so what the coach sees is exactly what the registry
can express — never what the user literally said.

Template coverage follows the tool's three primary uses:
  1. users under stress who need coping skills,
  2. users with a task/event they want to perform well on,
  3. users in distress (recognize, then help),
plus continuity glue (topics discussed, resources shared, preferences).
"""

import os
import sqlite3

from prompts.topics import content_topic_keys, TOPICS

_PROCESSED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "processed_resources")

# ------------------------------------------------------------- vocabularies

TOPIC_KEYS = content_topic_keys()

THEMES = [
    "stress_basics", "stress_physiology", "types_of_stress",
    "cognitive_appraisal", "coping_strategies", "burnout",
    "sleep_and_fatigue", "recovery_strategies", "adverse_events",
    "performance_under_pressure", "performance_zones", "mindset",
    "confidence", "emotional_intelligence", "team_resilience",
    "memory_and_study", "program_overview",
]

SKILLS = [
    "tactical_breathing", "deep_breathing", "pmr", "mindfulness",
    "meditation", "visualization", "self_talk", "goal_setting",
    "attention_control", "emotion_regulation", "distancing", "acceptance",
    "adaptability", "setback_management",
]

SKILL_REASONS = [
    "stress_management", "performance_preparation", "distress_relief",
    "sleep_and_recovery", "general_interest",
]

PRACTICE_OUTCOMES = [
    "helped", "somewhat_helped", "did_not_help", "found_it_difficult",
]

ASSESSMENTS = [
    "pss", "cope_inventory", "olbi", "umsat", "mindset_quiz", "ei_quiz",
    "izop",
]

BANDS = ["low", "moderate", "high", "not_scored"]

STRESSOR_AREAS = [
    "work", "school", "family", "relationships", "health", "sleep",
    "finances", "deployment", "major_life_change", "unspecified",
]

LEVELS = ["mild", "moderate", "high"]

EVENT_TYPES = [
    "presentation", "exam_or_test", "interview", "competition",
    "evaluation_or_inspection", "deployment_or_operation",
    "public_performance", "difficult_conversation", "deadline", "other",
]

TIMEFRAMES = ["today", "this_week", "next_week", "this_month", "later"]

CONCERNS = [
    "anxiety", "focus", "confidence", "preparation", "sleep", "motivation",
    "none_stated",
]

EVENT_OUTCOMES = [
    "went_well", "mixed", "went_poorly", "did_not_happen",
]

DISTRESS_INDICATORS = [
    "overwhelm", "anxiety", "low_mood", "anger_or_frustration",
    "sleep_problems", "burnout_signs", "grief_or_loss", "unspecified",
]

SUPPORT_ACTIONS = [
    "grounding_exercise", "breathing_exercise", "listened_and_validated",
    "shared_coping_resources", "suggested_professional_support",
    "made_follow_up_plan",
]

PREFERENCES = [
    "prefers_guided_audio", "prefers_video", "prefers_reading",
    "prefers_short_sessions", "prefers_step_by_step",
    "prefers_direct_answers", "prefers_gentle_tone",
    "prefers_scientific_detail",
]

GOAL_AREAS = [
    "stress_management", "sleep", "recovery", "performance",
    "skill_practice", "work_life_balance",
]

RESOURCE_PURPOSES = [
    "learn_concept", "guided_practice", "self_assessment", "reference",
]

# Display labels where de-underscoring isn't enough.
_LABELS = {
    "pss": "Perceived Stress Scale (PSS)",
    "cope_inventory": "COPE Inventory",
    "olbi": "Oldenburg Burnout Inventory (OLBI)",
    "umsat": "UMSAT mental-skills self-assessment",
    "mindset_quiz": "Mindset Quiz",
    "ei_quiz": "Emotional Intelligence Quiz",
    "izop": "IZOP zones worksheet",
    "pmr": "progressive muscle relaxation (PMR)",
    "self_talk": "self-talk",
    "none_stated": "none stated",
    "not_scored": "not scored / kept private",
}


def label(value) -> str:
    """Human-readable form of an enum value."""
    v = str(value or "")
    return _LABELS.get(v, v.replace("_", " "))


def topic_label(key: str) -> str:
    return TOPICS.get(key, {}).get("label", key)


# ---------------------------------------------------------------- templates
# slots: name -> {"enum": [...], "required": bool}  or  {"ref": "resource"}.
# Slot names are unique across templates so the record_memories tool can
# expose one flat union schema. "follow_up" marks templates the coach
# should proactively pick up in a later session.

TEMPLATES: dict[str, dict] = {
    "topic_discussed": {
        "description": "A subject area covered this session.",
        "slots": {
            "theme": {"enum": THEMES, "required": True},
            "topic": {"enum": TOPIC_KEYS, "required": False},
        },
        "follow_up": False,
    },
    "skill_introduced": {
        "description": "A mental skill explained or taught for the first time.",
        "slots": {
            "skill": {"enum": SKILLS, "required": True},
            "reason": {"enum": SKILL_REASONS, "required": False},
        },
        "follow_up": False,
    },
    "skill_practiced": {
        "description": "The user actually practised a skill during the session.",
        "slots": {
            "skill": {"enum": SKILLS, "required": True},
            "outcome": {"enum": PRACTICE_OUTCOMES, "required": True},
        },
        "follow_up": False,
    },
    "practice_commitment": {
        "description": "The user agreed to practise a skill on their own.",
        "slots": {
            "skill": {"enum": SKILLS, "required": True},
            "timeframe": {"enum": TIMEFRAMES, "required": False},
        },
        "follow_up": True,
    },
    "resource_shared": {
        "description": "A library resource was delivered to the user.",
        "slots": {
            "resource": {"ref": "resource", "required": True},
            "purpose": {"enum": RESOURCE_PURPOSES, "required": False},
        },
        "follow_up": True,
    },
    "assessment_result": {
        "description": "The user completed a self-assessment.",
        "slots": {
            "assessment": {"enum": ASSESSMENTS, "required": True},
            "band": {"enum": BANDS, "required": False},
        },
        "follow_up": False,
    },
    "upcoming_event": {
        "description": "The user is preparing for a demanding event or task.",
        "slots": {
            "event_type": {"enum": EVENT_TYPES, "required": True},
            "timeframe": {"enum": TIMEFRAMES, "required": False},
            "concern": {"enum": CONCERNS, "required": False},
        },
        "follow_up": True,
    },
    "event_outcome": {
        "description": "How a previously discussed event turned out.",
        "slots": {
            "event_type": {"enum": EVENT_TYPES, "required": True},
            "result": {"enum": EVENT_OUTCOMES, "required": True},
        },
        "follow_up": False,
    },
    "stress_reported": {
        "description": "The user reported ongoing stress in a broad life area.",
        "slots": {
            "area": {"enum": STRESSOR_AREAS, "required": True},
            "level": {"enum": LEVELS, "required": False},
        },
        "follow_up": False,
    },
    "distress_supported": {
        "description": "Signs of distress were recognized and support was given.",
        "slots": {
            "indicator": {"enum": DISTRESS_INDICATORS, "required": True},
            "action": {"enum": SUPPORT_ACTIONS, "required": True},
        },
        "follow_up": True,
    },
    "preference_noted": {
        "description": "A lasting preference about how the user likes to be coached.",
        "slots": {
            "preference": {"enum": PREFERENCES, "required": True},
        },
        "follow_up": False,
    },
    "goal_set": {
        "description": "The user set a goal to work toward.",
        "slots": {
            "goal_area": {"enum": GOAL_AREAS, "required": True},
            "timeframe": {"enum": TIMEFRAMES, "required": False},
        },
        "follow_up": True,
    },
}


# ---------------------------------------------------------------- rendering

def _resource_name(slots: dict) -> str:
    title = (slots.get("title") or "").strip()
    if title:
        return f"'{title}'"
    return f"resource #{slots.get('resource', '?')} ({slots.get('provider', '?')})"


def render_memory(template: str, slots: dict) -> str:
    """Deterministic human-readable line for a validated memory record."""
    s = slots
    if template == "topic_discussed":
        text = f"Discussed {label(s['theme'])}"
        if s.get("topic"):
            text += f" (topic area: {topic_label(s['topic'])})"
        return text + "."
    if template == "skill_introduced":
        text = f"Introduced the skill: {label(s['skill'])}"
        if s.get("reason"):
            text += f" (context: {label(s['reason'])})"
        return text + "."
    if template == "skill_practiced":
        return (f"User practised {label(s['skill'])} during the session — "
                f"it {label(s['outcome'])}.")
    if template == "practice_commitment":
        text = f"User agreed to practise {label(s['skill'])} on their own"
        if s.get("timeframe"):
            text += f" ({label(s['timeframe'])})"
        return text + ". Follow up on how it went."
    if template == "resource_shared":
        text = f"Shared the resource {_resource_name(s)}"
        if s.get("purpose"):
            text += f" for {label(s['purpose'])}"
        return text + ". Worth asking whether they got to it."
    if template == "assessment_result":
        text = f"User completed the {label(s['assessment'])}"
        band = s.get("band")
        if band and band != "not_scored":
            text += f" — result band: {label(band)}"
        return text + "."
    if template == "upcoming_event":
        text = f"User is preparing for a {label(s['event_type'])}"
        if s.get("timeframe"):
            text += f" ({label(s['timeframe'])})"
        if s.get("concern") and s["concern"] != "none_stated":
            text += f"; main concern: {label(s['concern'])}"
        return text + ". Ask how preparation (or the event itself) went."
    if template == "event_outcome":
        return (f"The {label(s['event_type'])} the user was preparing for "
                f"{label(s['result'])}.")
    if template == "stress_reported":
        level = label(s.get("level") or "ongoing")
        return f"User reported {level} stress related to {label(s['area'])}."
    if template == "distress_supported":
        return (f"User showed signs of distress ({label(s['indicator'])}); "
                f"support given: {label(s['action'])}. Check in gently.")
    if template == "preference_noted":
        return f"Coaching preference: {label(s['preference'])}."
    if template == "goal_set":
        text = f"User set a goal: {label(s['goal_area'])}"
        if s.get("timeframe"):
            text += f" ({label(s['timeframe'])})"
        return text + ". Ask about progress."
    return f"{template}: {slots}"


# --------------------------------------------------------------- validation

def _validate_resource_slots(slots: dict, errors: list) -> dict:
    """Validate the stored form of a resource reference:
    provider (known provider key) + resource (int id) + optional title."""
    out = {}
    provider = slots.get("provider")
    if provider not in TOPIC_KEYS:
        errors.append(f"unknown resource provider {provider!r}")
        return out
    try:
        out["resource"] = int(slots.get("resource"))
    except (TypeError, ValueError):
        errors.append(f"resource id must be an integer, got "
                      f"{slots.get('resource')!r}")
        return out
    out["provider"] = provider
    title = slots.get("title")
    if title is not None:
        out["title"] = str(title)[:200]
    return out


def validate_memory(template: str, slots: dict) -> tuple[dict, list[str]]:
    """Validate one memory record against the registry.

    Returns (clean_slots, errors). clean_slots contains only registered
    slot names with checked values — anything free-form is rejected, which
    is the privacy contract's enforcement point.
    """
    spec = TEMPLATES.get(template)
    if spec is None:
        return {}, [f"unknown template {template!r}"]
    slots = slots or {}
    errors: list[str] = []
    clean: dict = {}

    for name, rule in spec["slots"].items():
        value = slots.get(name)
        if rule.get("ref") == "resource":
            if slots.get("resource") is None and slots.get("provider") is None:
                if rule.get("required"):
                    errors.append(f"missing required resource reference")
                continue
            clean.update(_validate_resource_slots(slots, errors))
            continue
        if value is None or value == "":
            if rule.get("required"):
                errors.append(f"missing required slot '{name}'")
            continue
        if value not in rule["enum"]:
            errors.append(
                f"slot '{name}' must be one of {rule['enum']}, "
                f"got {value!r}")
            continue
        clean[name] = value

    allowed = set(spec["slots"]) | {"provider", "resource", "title"}
    extras = [k for k in slots if k not in allowed]
    if extras:
        errors.append(
            f"unknown slot(s) {extras} — free-form values are not storable")

    return clean, errors


def resolve_resource_title(provider: str, resource_id) -> str | None:
    """Look a resource title up in the topic's library database (used when a
    manually added or scenario-loaded record has no title snapshot)."""
    if provider not in TOPIC_KEYS:
        return None
    db_path = os.path.join(_PROCESSED_ROOT, provider, "database.db")
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT title FROM resources WHERE id = ?",
                           (int(resource_id),)).fetchone()
        conn.close()
        return row[0] if row else None
    except (sqlite3.Error, TypeError, ValueError):
        return None


# ------------------------------------------------------------- descriptions

def registry_for_api() -> dict:
    """JSON-friendly description of the registry (drives the memory
    browser's template-aware add/edit form)."""
    out = {}
    for key, spec in TEMPLATES.items():
        slots = {}
        for name, rule in spec["slots"].items():
            if rule.get("ref") == "resource":
                slots[name] = {
                    "ref": "resource",
                    "required": bool(rule.get("required")),
                    "providers": TOPIC_KEYS,
                }
            else:
                slots[name] = {
                    "enum": rule["enum"],
                    "required": bool(rule.get("required")),
                }
        out[key] = {
            "description": spec["description"],
            "follow_up": spec["follow_up"],
            "slots": slots,
        }
    return out


def registry_prompt_lines() -> list[str]:
    """One line per template, for the memory agent's tool description."""
    lines = []
    for key, spec in TEMPLATES.items():
        parts = []
        for name, rule in spec["slots"].items():
            req = "required" if rule.get("required") else "optional"
            if rule.get("ref") == "resource":
                parts.append(f"{name} ({req}: RES_xxxxx id from this session)")
            else:
                parts.append(f"{name} ({req})")
        lines.append(f"- {key}: {spec['description']} Slots: {', '.join(parts)}.")
    return lines


def slot_union_properties() -> dict:
    """JSON-schema properties for every slot name across all templates
    (the flat union the record_memories tool exposes)."""
    props: dict = {}
    for spec in TEMPLATES.values():
        for name, rule in spec["slots"].items():
            if name in props:
                continue
            if rule.get("ref") == "resource":
                props[name] = {
                    "type": "string",
                    "description": (
                        "Exact resource id retrieved this session, "
                        "e.g. 'RES_00004'."
                    ),
                }
            else:
                props[name] = {"type": "string", "enum": rule["enum"]}
    return props
