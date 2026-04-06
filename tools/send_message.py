from .base import BaseTool


class SendMessage(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Responds to the user with a message. This should be used to provide information "
                "to the user, or to ask them a question. This should be used as often as possible "
                "to keep the user informed and engaged. Supports markdown formatting."
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

        state["has_responded"] = True
        state["response_text"] += arguments["message"] + "\n"

        if chat_id in message_queues:
            message_queues[chat_id].put({
                "content": state["response_text"],
                "role": "assistant",
            })

        return "Queued message to user."
