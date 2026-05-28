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
]

TOOLSETS: dict[str, list[str]] = {
    "default": [
        "send_message",
        "search_resources",
        "examine_resource",
    ],
    # Used by the router profile and every per-topic profile.
    "topic_bot": _TOPIC_BOT_TOOLS,
}
