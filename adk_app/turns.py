"""
Per-turn context registry.

The tool plugins in ``tools/`` take a plain ``context`` dict (queues, turn
state, the embedding client, retrieved resources, the active library …).
Under ADK a tool is invoked with a ``ToolContext`` instead, so the dict now
lives here, keyed by the ADK session id the agent is running in: the coach's
chat session, or the scratch session a summarizer / memory-agent spawn runs
in. ``adk_app.tools`` looks the dict up on every call; ``adk_app.tracing``
uses it to route events to the right debug pane.

Each turn also owns a single-worker thread executor. The legacy tools are
blocking (HTTP to the embedding service, SQLite) so they must not run on
the event loop, and a single worker gives them the sequential semantics
they were written for — ADK executes a model response's tool calls in
parallel — while keeping the embedding client's SQLite connection on one
thread, as SQLite requires.
"""

from concurrent.futures import ThreadPoolExecutor

_turns: dict[str, dict] = {}


def new_context(session_id: str, agent: str, **fields) -> dict:
    """Build a tool context for one agent run and register it."""
    context = {
        "session_id": session_id,
        "agent": agent,
        "state": {"done": False, "has_responded": False, "response_text": ""},
        "existing_resources": [],
        "fields_to_remove": ["embedding"],
        # No live history list any more: tools that used to rewrite the
        # system message in place (switch_mode) see None and skip it; the
        # instruction provider recomputes the prompt from state instead.
        "conversation_history": None,
        "tools_in_response": [],
        "tool_calls": 0,
        "last_text": "",
        "executor": fields.pop("executor", None) or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"botbase-tools-{agent}"),
    }
    context.update(fields)
    _turns[session_id] = context
    return context


def get(session_id: str) -> dict:
    try:
        return _turns[session_id]
    except KeyError:
        raise RuntimeError(
            f"No BotBase turn context registered for ADK session "
            f"{session_id!r}; agents must be run through adk_app.runtime."
        ) from None


def find(session_id: str) -> dict | None:
    return _turns.get(session_id)


def release(session_id: str) -> None:
    _turns.pop(session_id, None)
