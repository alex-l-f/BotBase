"""
Turn runtime: the API ``server.py`` calls, backed by an ADK ``Runner``.

A chat is an ADK session (session id = chat id) in one session service,
so the conversation history the old loop rebuilt from the client's
``fullContext`` on every request now lives server-side; the client's copy
is only used to seed a session the server doesn't have yet (a restart, or
the simulator resuming). Each turn builds the coach agent for the chat's
profile / architecture, runs one invocation on the shared event loop,
renders the new events back into the message list the frontend renders,
and — in the multi architecture with an active user profile — hands the
turn to the memory agent, the store's single writer.

The server is synchronous Flask; ADK is asyncio. One background event loop
serves every turn (``run_coroutine_threadsafe``), which keeps LiteLLM's
per-loop client cache and ADK's per-loop executors stable across requests.
"""

import asyncio
import logging
import threading
import uuid
from queue import Queue
from typing import Dict, List

from google.adk.agents.run_config import RunConfig
from google.adk.sessions import InMemorySessionService

from embedding_client import EmbeddingSearchClient
from memory import MemoryStore
from prompts import get_toolset
from prompts.topics import TOPICS
from tools import load_tools
from trace import AgentTracer, clip

from . import history, turns
from .agents import COACH_AGENT, COACH_MAX_LLM_CALLS, build_coach, make_runner
from .models import make_model
from .subagents import run_memory_agent, run_until_done

log = logging.getLogger(__name__)

APP_NAME = "botbase"

load_tools()

# Global message queues and chat status tracking (read by server.py).
message_queues: dict[str, Queue] = {}
chat_status: dict[str, dict] = {}

# Profile-scoped template memory + instrumentation store. Single-writer:
# add_memories() is called from exactly one place (the memory agent's
# output at the end of get_LM_response); everything else only reads.
memory_store = MemoryStore()

session_service = InMemorySessionService()

# The one event loop every ADK run uses, on its own daemon thread.
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, name="botbase-adk-loop",
                 daemon=True).start()


def _run(coro):
    """Run a coroutine on the ADK loop from a synchronous caller."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def _initial_database_for_profile(profile: str | None) -> str | None:
    """Pick the search-provider key the turn should start with.

    Topic profiles point at their per-topic index. The router has no library.
    The legacy 'default' profile keeps using the original 'imported' index
    so existing demo setups don't break.
    """
    if profile in TOPICS:
        return TOPICS[profile].get("provider")
    return "imported"


def create_chat_session() -> str:
    chat_id = str(uuid.uuid4())
    message_queues[chat_id] = Queue()
    chat_status[chat_id] = {"is_complete": False}
    return chat_id


def get_messages(chat_id: str) -> list:
    if chat_id not in message_queues:
        return []
    messages = []
    while not message_queues[chat_id].empty():
        messages.append(message_queues[chat_id].get())
    return messages


def is_chat_complete(chat_id: str) -> bool:
    return chat_id in chat_status and chat_status[chat_id]["is_complete"]


def reset_complete(chat_id: str) -> None:
    if chat_id in chat_status:
        chat_status[chat_id]["is_complete"] = False


async def _ensure_session(chat_id: str, user_id: int | None):
    """The chat's ADK session, created on first use.

    ADK sessions belong to a user id fixed at creation; it is derived from
    the memory profile on the first turn and pinned in chat_status so
    later turns address the same session.
    """
    status = chat_status.setdefault(chat_id, {"is_complete": False})
    adk_user = status.get("adk_user") or (
        f"profile:{user_id}" if user_id else "anonymous")
    status["adk_user"] = adk_user
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=adk_user, session_id=chat_id)
    if session is None:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=adk_user, session_id=chat_id)
    return session


def get_LM_response(conversation_dict: List[Dict[str, str]], chat_id: str,
                    model: str = None, system_prompt: str = None,
                    toolset=None, profile: str = None, arch: str = "single",
                    user_id: int | None = None):
    """Run one coach turn. Returns (response_text, new_messages, full_context)."""
    return _run(_turn(conversation_dict, chat_id, model, system_prompt,
                      toolset, profile, arch, user_id))


async def _turn(conversation_dict, chat_id, model, system_prompt, toolset,
                profile, arch, user_id):
    if toolset is None:
        toolset = get_toolset(profile or "default", arch)

    last_user_message = None
    for msg in reversed(conversation_dict or []):
        if msg.get("role") == "user":
            last_user_message = msg.get("content")
            break
    if not last_user_message:
        raise ValueError("A turn needs a user message to respond to.")

    session = await _ensure_session(chat_id, user_id)
    adk_user = chat_status[chat_id]["adk_user"]
    if not session.events and len(conversation_dict) > 1:
        # The client holds history this server has never seen: adopt it.
        await history.seed_session(session_service, session,
                                   conversation_dict[:-1], COACH_AGENT)
    events_before = len(session.events)

    tracer = AgentTracer(chat_id, message_queues, memory_store)

    # The one always-injected memory tier: a small, hard-capped snapshot of
    # the active profile's structured memory notes. Multi-agent only — the
    # single-agent baseline stays a pure comparison target.
    profile_block = ""
    if arch == "multi" and user_id:
        profile_block = memory_store.profile_block(user_id, exclude_chat=chat_id)
        if profile_block:
            tracer.emit("memory", "inject", {"chars": len(profile_block)},
                        persist={"block": profile_block})

    embedding_search = EmbeddingSearchClient()
    turn = turns.new_context(
        chat_id, COACH_AGENT,
        message_queues=message_queues,
        chat_id=chat_id,
        chat_status=chat_status,
        last_user_message=last_user_message,
        embedding_search=embedding_search,
        database=_initial_database_for_profile(profile),
        arch=arch,
        profile=profile,
        system_prompt=system_prompt,
        tracer=tracer,
        memory_store=memory_store,
        user_id=user_id,
        profile_block=profile_block,
    )
    state = turn["state"]

    try:
        runner = make_runner(build_coach(make_model(model), toolset),
                             session_service, APP_NAME)
        # Topic profiles carry their mode in session state so the
        # instruction provider (and switch_mode) can read and update it.
        state_delta = {"mode": profile} if profile in TOPICS else None

        tracer.emit(COACH_AGENT, "turn_start", {
            "arch": arch,
            "user_message": clip(last_user_message, 300),
        })

        await run_until_done(
            runner, user_id=adk_user, session_id=chat_id,
            text=last_user_message, state_delta=state_delta,
            run_config=RunConfig(max_llm_calls=COACH_MAX_LLM_CALLS))
        if not state["has_responded"]:
            # The model answered in plain text, which the user never sees.
            await run_until_done(
                runner, user_id=adk_user, session_id=chat_id,
                text=("Reminder: nothing you write as plain text reaches "
                      "the user. Deliver your reply now with the "
                      "send_message tool."),
                run_config=RunConfig(max_llm_calls=3,
                                     custom_metadata=dict(history.NUDGE_METADATA)))

        session = await session_service.get_session(
            app_name=APP_NAME, user_id=adk_user, session_id=chat_id)
        full_context = history.events_to_messages(session.events)
        new_messages = history.events_to_messages(
            session.events[events_before:], include_user=False)

        # Memory extraction — the single writer's only call site. The memory
        # agent turns this turn's conversational surface into anonymous
        # template records (structured generation via record_memories); only
        # what fits the template registry can be stored. Runs only in the
        # multi architecture with an active user profile.
        if arch == "multi" and user_id:
            turn_messages = [{"role": "user", "content": last_user_message}]
            turn_messages.extend(new_messages)
            try:
                prior_notes = memory_store.list_memories(
                    user_id, limit=20)["memories"]
                records = await run_memory_agent(turn_messages, turn, prior_notes)
                stored = memory_store.add_memories(user_id, chat_id, records) \
                    if records else []
                tracer.emit("memory", "write", {
                    "rows": len(stored),
                    "profile": user_id,
                }, persist={"stored": stored})
            except Exception as exc:
                log.warning("Memory extraction failed: %s", exc)

        tracer.emit(COACH_AGENT, "turn_done", {})

        if chat_id in chat_status:
            chat_status[chat_id]["is_complete"] = True

        return state["response_text"], new_messages, full_context
    finally:
        turns.release(chat_id)
        executor = turn["executor"]
        # The embedding client's SQLite connection belongs to the executor
        # thread, so it is closed there.
        await asyncio.get_running_loop().run_in_executor(
            executor, embedding_search.close)
        executor.shutdown(wait=False)
