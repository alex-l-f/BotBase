"""
Per-agent instrumentation for the multi-agent architecture.

The tracer does two things with every event:
1. Pushes a clipped copy into the chat's message queue so the frontend's
   debug panel can stream it live into the right agent sub-pane.
2. Persists the full event to memory.db's trace_log via the MemoryStore
   (the single writer) — that log is the eval corpus and the future
   push-channel specification (brief §6, "log every tool call, in and out").
"""

import json
from datetime import datetime


def clip(value, limit: int = 600) -> str:
    """Render a value as a display string, truncated for queue transport."""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = str(value)
    if len(value) > limit:
        return value[:limit] + " …[truncated]"
    return value


class AgentTracer:
    def __init__(self, chat_id: str, message_queues: dict, memory_store=None):
        self.chat_id = chat_id
        self.message_queues = message_queues
        self.memory_store = memory_store

    def emit(self, agent: str, event_type: str, data: dict | None = None,
             persist=None):
        """Stream an event to the debug UI and persist it.

        data     — small, clipped payload for the live queue.
        persist  — optional full payload for the trace log; defaults to data.
        """
        event = {
            "role": "agent_event",
            "agent": agent,
            "type": event_type,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "data": data or {},
        }
        queue = self.message_queues.get(self.chat_id)
        if queue is not None:
            queue.put(event)
        if self.memory_store is not None:
            try:
                self.memory_store.log_trace(
                    self.chat_id, agent, event_type,
                    persist if persist is not None else (data or {}),
                )
            except Exception:
                # Instrumentation must never take down the turn.
                pass


class NullTracer:
    """Drop-in no-op so call sites don't need None checks."""

    def emit(self, agent, event_type, data=None, persist=None):
        pass
