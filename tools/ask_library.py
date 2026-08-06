from .base import BaseTool


class AskLibrary(BaseTool):
    """Coach-side entry point to the library summarizer sub-agent.

    Spawns a fresh read-only research agent over the active topic library
    and returns its structured summary. The one tool the brief's v0 table
    calls `ask_repo(question) -> summary`.
    """

    schema = {
        "type": "function",
        "function": {
            "name": "ask_library",
            "description": (
                "Delegate a research question to the library summarizer — a "
                "read-only agent that searches the active topic's resource "
                "library and returns a structured summary (answer, "
                "key_points, resources with ids, confidence, source, "
                "timestamp). The resource ids it returns work directly with "
                "provide_file and open_course_page. The summarizer cannot "
                "see the conversation, so the question must be "
                "self-contained. Most turns need at most one call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "A specific, self-contained research question, "
                            "including anything from the conversation the "
                            "summarizer needs to answer it well."
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        state = context["state"]
        state["done"] = False

        question = (arguments.get("question") or "").strip()
        if not question:
            return "ERROR: Missing 'question' argument in ask_library."

        if not context.get("database"):
            return (
                "ERROR: No topic library is active. Use switch_mode to "
                "select a topic before asking the library."
            )

        # Imported lazily: tools are auto-discovered at import time, and the
        # summarizer module pulls in prompts + the tool registry itself.
        from summarizer_agent import run_summarizer

        return run_summarizer(question, context)
