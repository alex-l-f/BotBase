from .base import BaseTool


class FinishTurn(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "finish_turn",
            "description": (
                "Ends the current turn and sends your response to the user. The user will then "
                "be able to reply to you. This should be called as often as reasonable, to ensure "
                "the user is kept up to date and can respond to your messages."
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
        if state["has_responded"]:
            state["done"] = True
            return "Message(s) sent to user. Waiting for reply."
        else:
            return (
                "ERROR: Must respond to user at least once. Make sure to send your message "
                "BEFORE calling the finish turn function. The user cannot see this error."
            )
