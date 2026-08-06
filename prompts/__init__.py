import importlib
import logging

from .profiles import PROFILES
from .topics import TOPICS

log = logging.getLogger(__name__)

_DEFAULT_PROFILE = "default"

# Agent architectures. "single" is the original one-agent librarian;
# "multi" is the coach/summarizer/memory split (multi-agent-paradigms-2026.md §6).
ARCHITECTURES = ("single", "multi")


def get_prompt(profile_name: str | None = None, arch: str = "single") -> str:
    """Return the system prompt string for *profile_name*.

    Falls back to the default profile when *profile_name* is None or unknown.
    When arch == "multi", topic profiles get the coach overlay appended so
    the prompt matches the delegation toolset.
    """
    name = profile_name or _DEFAULT_PROFILE
    if name not in PROFILES:
        log.warning("Unknown profile %r — falling back to default", name)
        name = _DEFAULT_PROFILE

    module_name = PROFILES[name]["prompt_module"]
    module = importlib.import_module(f".{module_name}", package=__name__)
    prompt = module.PROMPT
    if arch == "multi" and name in TOPICS:
        from .coach_overlay import overlay_for
        prompt = prompt + overlay_for(name)
    return prompt


def get_toolset(profile_name: str | None = None, arch: str = "single") -> str | None:
    """Return the toolset key for *profile_name* (for use with tools.get_schemas)."""
    name = profile_name or _DEFAULT_PROFILE
    if name not in PROFILES:
        log.warning("Unknown profile %r — falling back to default", name)
        name = _DEFAULT_PROFILE
    toolset = PROFILES[name].get("toolset")
    if arch == "multi" and toolset == "topic_bot":
        return "coach"
    return toolset


def list_profiles() -> list[str]:
    """Return all registered profile names."""
    return list(PROFILES.keys())
