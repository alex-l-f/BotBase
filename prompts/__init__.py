import importlib
import logging

from .profiles import PROFILES

log = logging.getLogger(__name__)

_DEFAULT_PROFILE = "default"


def get_prompt(profile_name: str | None = None) -> str:
    """Return the system prompt string for *profile_name*.

    Falls back to the default profile when *profile_name* is None or unknown.
    """
    name = profile_name or _DEFAULT_PROFILE
    if name not in PROFILES:
        log.warning("Unknown profile %r — falling back to default", name)
        name = _DEFAULT_PROFILE

    module_name = PROFILES[name]["prompt_module"]
    module = importlib.import_module(f".{module_name}", package=__name__)
    return module.PROMPT


def get_toolset(profile_name: str | None = None) -> str | None:
    """Return the toolset key for *profile_name* (for use with tools.get_schemas)."""
    name = profile_name or _DEFAULT_PROFILE
    if name not in PROFILES:
        log.warning("Unknown profile %r — falling back to default", name)
        name = _DEFAULT_PROFILE
    return PROFILES[name].get("toolset")


def list_profiles() -> list[str]:
    """Return all registered profile names."""
    return list(PROFILES.keys())
