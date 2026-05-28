from .base import BaseTool


class FinishTurn(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "finish_turn",
            "description": (
                "Ends the current turn and hands control back to the user. "
                "IMPORTANT: This tool must be called ALONE, in a response "
                "that contains no other tool calls. Do not call it in the "
                "same response as send_message, search_resources, "
                "switch_mode, etc. The correct flow is: (1) make your "
                "tool calls, (2) read the tool results, (3) if there is "
                "nothing left to do this turn, issue a fresh response "
                "containing only finish_turn. Calling finish_turn next to "
                "other tools will fail."
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

        # Refuse when other tools were called in the same model response.
        # The agent loop populates `tools_in_response` with the full list
        # of calls produced by the LLM this iteration; we error if any
        # sibling exists so the model is forced to wait for the prior
        # tool results before deciding the turn is over.
        siblings = [
            t for t in context.get("tools_in_response") or []
            if t.get("name") != "finish_turn"
        ]
        if siblings:
            sibling_names = sorted({t.get("name", "?") for t in siblings})
            return (
                "ERROR: finish_turn cannot be called alongside other tools. "
                f"This response also contained: {', '.join(sibling_names)}. "
                "Wait for those tools' results, decide what (if anything) "
                "still needs to happen, then in a separate response call "
                "finish_turn on its own."
            )

        if state["has_responded"]:
            state["done"] = True
            return "Message(s) sent to user. Waiting for reply."
        return (
            "ERROR: Must respond to user at least once. Make sure to send "
            "your message BEFORE calling the finish turn function. The "
            "user cannot see this error."
        )
