from .base import BaseTool
from prompts.topics import TOPICS, ROUTER_MODE, content_topic_keys


def _load_prompt(prompt_module: str) -> str:
    """Import the prompt module and return its PROMPT string."""
    import importlib
    module = importlib.import_module(f"prompts.{prompt_module}")
    return module.PROMPT


class SwitchMode(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "switch_mode",
            "description": (
                "Switch the chatbot to a different topic mode. Each mode has "
                "its own focus area, prompt, and resource library. Switching "
                "rebuilds the search context to point at the new topic's "
                "library and updates how you should approach the conversation. "
                "Call this when the user's needs no longer fit the current "
                "topic. After calling, briefly acknowledge the switch to the "
                "user, then continue helping them in the new mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_mode": {
                        "type": "string",
                        "enum": list(TOPICS.keys()),
                        "description": (
                            "The mode to switch to. Use 'router' if you need "
                            "to ask the user which topic fits best. Otherwise "
                            "pick the most relevant content topic."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Brief reason for the switch (1 sentence). Used "
                            "for logging and to seed your acknowledgement to "
                            "the user."
                        ),
                    },
                },
                "required": ["target_mode"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        state = context["state"]
        state["done"] = False

        target = arguments.get("target_mode")
        if not target:
            return "ERROR: Missing 'target_mode' argument."
        if target not in TOPICS:
            valid = ", ".join(TOPICS.keys())
            return f"ERROR: Unknown mode {target!r}. Valid modes: {valid}"

        topic = TOPICS[target]

        # Load the new prompt and swap it into the live conversation history.
        # context['conversation_history'] is a reference to the same list the
        # Conversation object holds, so this takes effect for the rest of the
        # current turn.
        try:
            new_prompt = _load_prompt(topic["prompt_module"])
        except Exception as exc:
            return f"ERROR: Could not load prompt for mode {target!r}: {exc}"

        history = context.get("conversation_history")
        if history and history[0].get("role") == "system":
            history[0]["content"] = new_prompt

        # Switch the active search provider so search_resources hits the
        # right index. None means the router (no content library).
        new_provider = topic["provider"]
        if new_provider is not None:
            context["database"] = new_provider
        else:
            # Router mode has no library — leave database as a sentinel.
            context["database"] = None

        # Resources collected before the switch are from a different topic
        # and will only confuse subsequent ranking, so reset.
        context["existing_resources"] = []

        # Persist the new mode across turns. chat_status is the per-chat
        # server-side state owned by agent.py; the chat-profile endpoint
        # reads from it on each new request.
        chat_status = context.get("chat_status")
        chat_id = context.get("chat_id")
        if chat_status is not None and chat_id in chat_status:
            chat_status[chat_id]["mode"] = target

        # Push a structured event into the message queue so the frontend can
        # update its mode indicator without having to poll a separate route.
        message_queues = context.get("message_queues") or {}
        if chat_id in message_queues:
            message_queues[chat_id].put({
                "role": "system_event",
                "event": "mode_switch",
                "mode": target,
                "label": topic["label"],
            })

        reason = arguments.get("reason") or ""
        valid_content_modes = ", ".join(content_topic_keys())
        return (
            f"Switched to mode '{target}' ({topic['label']}). "
            f"Search now scoped to provider '{new_provider}'. "
            f"Reason: {reason}. "
            f"Available content modes: {valid_content_modes}."
        )
