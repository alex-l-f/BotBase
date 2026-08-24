# Toolset definitions — map a name to the list of tool names it exposes.
# Tool names must match the "name" field inside each tool's schema.

# Tools shared by every topic-aware profile (router + 5 content topics).
# Switching modes does not change the toolset; only the prompt and the
# active database/provider change.
_TOPIC_BOT_TOOLS: list[str] = [
    "send_message",
    "search_resources",
    "examine_resource",
    "switch_mode",
    "provide_file",
    "open_course_page",
]

# Multi-agent architecture (multi-agent-paradigms-2026.md §6).
# The coach is the orchestrator and the only user-facing agent: it keeps
# every action/delivery tool (writes stay single-threaded) but delegates
# library research to the summarizer via ask_library and recalls past
# sessions via memory_search. It does NOT get raw search tools.
_COACH_TOOLS: list[str] = [
    "send_message",
    "ask_library",
    "memory_search",
    "switch_mode",
    "provide_file",
    "open_course_page",
    "finish_turn",
]

# The summarizer is a read-only explorer: search + examine, and the
# structured-output contract it must finish with. No user-facing tools.
_SUMMARIZER_TOOLS: list[str] = [
    "search_resources",
    "examine_resource",
    "return_summary",
]

# The memory agent's whole tool surface is its structured output contract:
# it can only speak in memory templates, which is the privacy guarantee.
_MEMORY_AGENT_TOOLS: list[str] = [
    "record_memories",
]

TOOLSETS: dict[str, list[str]] = {
    "default": [
        "send_message",
        "search_resources",
        "examine_resource",
    ],
    # Used by the router profile and every per-topic profile (single-agent).
    "topic_bot": _TOPIC_BOT_TOOLS,
    # Multi-agent: coach (orchestrator) and summarizer (read-only sub-agent).
    "coach": _COACH_TOOLS,
    "summarizer": _SUMMARIZER_TOOLS,
    "memory_agent": _MEMORY_AGENT_TOOLS,
}
