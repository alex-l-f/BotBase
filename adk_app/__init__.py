"""
BotBase on Google's Agent Development Kit (ADK).

The three agents of the multi-agent architecture — coach, summarizer,
memory — are ADK ``LlmAgent``s; the hand-rolled LLM-in-a-loop, tool
dispatch, conversation history and backend adapters they used to run on
are replaced by ADK's ``Runner``, session service, function-calling flow
and ``LiteLlm`` model wrapper. Everything domain-specific (the tools in
``tools/``, the prompts, the template memory store, the tracer) is reused
unchanged: ``adk_app.tools`` adapts the existing tool plugins to ADK's
``BaseTool`` and ``adk_app.turns`` carries the per-turn context they expect.

``adk_app.runtime`` exposes the same API ``server.py`` consumed from the
old ``agent`` module, so the Flask server, frontend and simulator are
untouched.
"""

from .models import BACKENDS, get_active_backend, set_backend
from .runtime import (
    chat_status,
    create_chat_session,
    get_LM_response,
    get_messages,
    is_chat_complete,
    memory_store,
    message_queues,
    reset_complete,
)

__all__ = [
    "BACKENDS",
    "chat_status",
    "create_chat_session",
    "get_LM_response",
    "get_active_backend",
    "get_messages",
    "is_chat_complete",
    "memory_store",
    "message_queues",
    "reset_complete",
    "set_backend",
]
