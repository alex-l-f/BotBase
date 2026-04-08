# Toolset definitions — map a name to the list of tool names it exposes.
# Tool names must match the "name" field inside each tool's schema.

TOOLSETS: dict[str, list[str]] = {
    "default": [
        "send_message",
        "finish_turn",
        "search_resources",
        "examine_resource",
    ],
    # Example: add custom toolsets for different bot profiles
    # "my_bot": [
    #     "send_message",
    #     "finish_turn",
    #     "my_custom_tool",
    # ],
}
