from .base import BaseTool


class MemorySearch(BaseTool):
    """Coach-side recall over the active profile's structured memory notes.

    Read-only: the MemoryStore is written from exactly one place (the
    memory agent's output at the end of each coach turn), never from a
    tool. Recall is a runtime decision by the coach, not an injection —
    the pull path from the brief's v0 table.
    """

    schema = {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Keyword search over the structured memory notes from this "
                "user's past sessions (skills practised, resources shared, "
                "assessments taken, upcoming events, follow-ups). Use it "
                "when the user refers to something from before, or when "
                "their history would materially improve your coaching. "
                "Notes are anonymized templates with timestamps — evidence "
                "that something happened, never a transcript. Finding "
                "nothing is a normal outcome."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Keywords to look for (e.g. 'breathing "
                            "practice', 'burnout assessment'). Plain words "
                            "work best."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        state = context["state"]
        state["done"] = False

        query = (arguments.get("query") or "").strip()
        if not query:
            return "ERROR: Missing 'query' argument in memory_search."

        store = context.get("memory_store")
        if store is None:
            return "ERROR: Memory is not available in this deployment."

        profile_id = context.get("user_id")
        if not profile_id:
            return (
                "No user profile is active in this session, so there is no "
                "past-session memory to search."
            )

        # Exclude the live chat — its content is already in context.
        hits = store.search(profile_id, query, limit=5,
                            exclude_chat=context.get("chat_id"))

        tracer = context.get("tracer")
        if tracer:
            tracer.emit("memory", "search",
                        {"query": query, "hits": len(hits)},
                        persist={"query": query, "hits": hits})

        if not hits:
            return (
                "No matching notes in this user's past-session memory. That "
                "is a normal outcome — do not invent a memory."
            )
        return {"hits": hits}
