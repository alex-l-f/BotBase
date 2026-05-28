"""
Single source of truth for the chatbot's topic modes.

Each topic ties together:
  - a stable mode key (used by switch_mode and as the profile name)
  - the human-readable label that appears in summaries and to the user
  - the embedding-service provider key (the directory under
    processed_resources/ that holds the per-topic index)
  - the prompt module (file under prompts/) that holds the topic's prompt

The profiles registry in prompts/profiles.py is built from this dict, so
adding a new mode here automatically wires it into the agent.
"""

ROUTER_MODE = "router"

TOPICS: dict[str, dict] = {
    ROUTER_MODE: {
        "label": "Router",
        "short_description": (
            "Triage. Identifies which topic mode best fits the user's needs "
            "and switches to it."
        ),
        "provider": None,
        "prompt_module": "router",
    },
    "coping_mental_skills": {
        "label": "Coping — Mental Skills",
        "short_description": (
            "Trainable mental-skill techniques: tactical breathing, PMR, "
            "mindfulness, visualization, self-talk, goal setting, attention "
            "control, emotion regulation, distancing, acceptance, "
            "adaptability, setback management."
        ),
        "provider": "coping_mental_skills",
        "prompt_module": "topic_coping_mental_skills",
    },
    "coping_recovery": {
        "label": "Coping — Recovery",
        "short_description": (
            "Recovery from stress: sleep, detachment from work, burnout "
            "(OLBI), recovery after adverse events, daylight-saving impact "
            "on health."
        ),
        "provider": "coping_recovery",
        "prompt_module": "topic_coping_recovery",
    },
    "coping_stress": {
        "label": "Coping — Stress",
        "short_description": (
            "Stress fundamentals: physiology, fight/flight/freeze, the "
            "amygdala, cognitive appraisal, problem- vs emotion-focused "
            "coping, PSS and COPE assessments."
        ),
        "provider": "coping_stress",
        "prompt_module": "topic_coping_stress",
    },
    "other_content": {
        "label": "R2MR Overview & Sleep",
        "short_description": (
            "R2MR program overview (history, courses, evidence base) and "
            "WRAIR sleep / fatigue guidance for shift workers and military "
            "medical personnel."
        ),
        "provider": "other_content",
        "prompt_module": "topic_other_content",
    },
    "performance": {
        "label": "Performance",
        "short_description": (
            "Performing under pressure: Optimized Performance Cycle, IZOP/SZOP "
            "zones, mental toughness vs. resilience, mindset, emotional "
            "intelligence, team resilience, performance coaching."
        ),
        "provider": "performance",
        "prompt_module": "topic_performance",
    },
}


def topic_keys() -> list[str]:
    """All topic mode keys, including 'router'."""
    return list(TOPICS.keys())


def content_topic_keys() -> list[str]:
    """Topic keys that have a real content provider (everything except router)."""
    return [k for k, v in TOPICS.items() if v["provider"] is not None]


def get_topic(key: str) -> dict | None:
    return TOPICS.get(key)


def topic_summary_lines() -> list[str]:
    """One-line topic descriptions, suitable for embedding in a prompt."""
    lines = []
    for key in content_topic_keys():
        t = TOPICS[key]
        lines.append(f"- `{key}` — **{t['label']}**: {t['short_description']}")
    return lines
