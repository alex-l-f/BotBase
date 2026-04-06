# Profile definitions — map a profile name to a prompt module and toolset.
# Add a new entry here for each variant that needs a different prompt/tool combination.
# - prompt_module: filename (without .py) inside prompts/ that contains a PROMPT string
# - toolset: key from tools/toolsets.py (or None to use all registered tools)

PROFILES: dict[str, dict] = {
    "default": {
        "prompt_module": "default",
        "toolset": "default",
    },
    # Example: a custom variant with its own prompt and a reduced tool set
    # "my_bot": {
    #     "prompt_module": "my_bot_prompt",
    #     "toolset": "my_bot_tools",
    # },
}
