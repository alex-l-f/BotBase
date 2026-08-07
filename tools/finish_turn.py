from .base import BaseTool


class FinishTurn(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "finish_turn",
            "description": (
                "Ends the current turn and hands control back to the user. "
                "It MAY be called in the same response as send_message — "
                "the message is delivered first, then the turn ends. It must "
                "NOT be combined with any other tool (search_resources, "
                "ask_library, switch_mode, provide_file, etc.): make those "
                "calls, read their results, and only then — when nothing is "
                "left to do — finish the turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        state = context["state"]

        # Refuse when tools other than send_message were called in the same
        # model response. send_message is exempt because its result carries
        # nothing the model needs to read — and forcing an extra round trip
        # to end the turn makes small models replay their previous response,
        # duplicating the message. Every other tool returns information that
        # should be seen before deciding the turn is over.
        siblings = [
            t for t in context.get("tools_in_response") or []
            if t.get("name") not in ("finish_turn", "send_message")
        ]
        if siblings:
            sibling_names = sorted({t.get("name", "?") for t in siblings})
            return (
                "ERROR: finish_turn cannot be called alongside other tools "
                f"(except send_message). This response also contained: "
                f"{', '.join(sibling_names)}. Wait for those tools' results, "
                "decide what (if anything) still needs to happen, then call "
                "finish_turn."
            )

        if state["has_responded"]:
            state["done"] = True
            return "Message(s) sent to user. Waiting for reply."
        return (
            "ERROR: Must respond to user at least once. Make sure to send "
            "your message BEFORE calling the finish turn function. The "
            "user cannot see this error."
        )
