import re
from difflib import SequenceMatcher

from .base import BaseTool

# How similar (0..1) a new message must be to an already-sent one to be
# suppressed as a repeat. Repeats observed in practice are near-verbatim
# with minor wording drift, which lands well above this; genuinely new
# content (even on the same subject) lands well below.
SIMILARITY_THRESHOLD = 0.90


def _normalize(text: str) -> str:
    """Collapse whitespace and case so trivial reformatting doesn't dodge
    the repeat check."""
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _similarity(a: str, b: str) -> float:
    """Longest-matching-subsequence similarity of two normalized strings.

    The quick_ratio tiers are upper bounds that avoid the full quadratic
    comparison when the pair obviously can't reach the threshold.
    """
    m = SequenceMatcher(None, a, b)
    if m.real_quick_ratio() < SIMILARITY_THRESHOLD:
        return 0.0
    if m.quick_ratio() < SIMILARITY_THRESHOLD:
        return 0.0
    return m.ratio()


class SendMessage(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Responds to the user with a message. This should be used to provide information "
                "to the user, or to ask them a question. This should be used as often as possible "
                "to keep the user informed and engaged. Supports markdown formatting. "
                "Send each distinct message exactly once — resending the same or a "
                "near-identical message is suppressed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the user. Supports markdown formatting.",
                    }
                },
                "required": ["message"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        if "message" not in arguments:
            return "ERROR: Missing 'message' argument in send_message command"

        state = context["state"]
        chat_id = context["chat_id"]
        message_queues = context["message_queues"]

        # Repeat guard: small models often re-emit their previous message
        # (verbatim or lightly reworded) instead of ending the turn. Catch
        # near-duplicates of anything already sent THIS turn and refuse to
        # deliver them, telling the model how to move on instead.
        normalized = _normalize(arguments["message"])
        sent = state.setdefault("sent_messages", [])
        for prior in sent:
            score = _similarity(normalized, prior)
            if score >= SIMILARITY_THRESHOLD:
                return (
                    f"SUPPRESSED: this message is {round(score * 100)}% identical to one "
                    "you already sent this turn, so it was NOT delivered. Do not repeat "
                    "yourself. Say something genuinely new, or if you have nothing to "
                    "add, call finish_turn now."
                )
        sent.append(normalized)

        state["has_responded"] = True
        state["response_text"] += arguments["message"] + "\n"

        if chat_id in message_queues:
            message_queues[chat_id].put({
                "content": state["response_text"],
                "role": "assistant",
            })

        return (
            "Message delivered to the user. Do not send this message (or a "
            "rephrasing of it) again. When there is nothing left to do this "
            "turn, call finish_turn."
        )
