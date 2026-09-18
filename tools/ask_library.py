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
        # The summarizer is an ADK sub-run, which has to be awaited from the
        # agent's event loop; adk_app.tools.AskLibraryTool does that and
        # reuses this schema. Only that path is wired into the coach.
        return (
            "ERROR: ask_library is only available through the ADK runtime "
            "(adk_app.tools.AskLibraryTool)."
        )
