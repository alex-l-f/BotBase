# Profile definitions — map a profile name to a prompt module and toolset.
# - prompt_module: filename (without .py) inside prompts/ that contains a PROMPT string
# - toolset: key from tools/toolsets.py (or None to use all registered tools)
#
# The R2MR topic profiles (router + 5 content topics) are generated from
# prompts/topics.py so adding a new topic in one place wires it everywhere.

from .topics import TOPICS

PROFILES: dict[str, dict] = {
    "default": {
        "prompt_module": "default",
        "toolset": "default",
    },
}

for _key, _topic in TOPICS.items():
    PROFILES[_key] = {
        "prompt_module": _topic["prompt_module"],
        "toolset": "topic_bot",
    }
