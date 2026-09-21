"""
The summarizer and memory sub-agent runs.

Both follow the same shape the old modules had — build a request from the
parent turn, run the agent until its terminal tool fires (``return_summary``
/ ``record_memories``), nudge once if it never does, then read the result
back out of the run's state — now on an ADK ``Runner`` with a fresh
in-memory session per spawn. That per-spawn session is the context
isolation the brief asks for: none of the corpus traffic enters the coach's
history. The parent's tracer, embedding client and executor are shared so
events land in the right debug pane and searches stay sequential.
"""

import json
import logging
import os
import uuid
from datetime import datetime

from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.sessions import InMemorySessionService
from google.genai import types

from trace import NullTracer, clip

from . import turns
from .agents import (
    MEMORY_AGENT,
    MEMORY_MAX_LLM_CALLS,
    SUMMARIZER_AGENT,
    SUMMARIZER_MAX_LLM_CALLS,
    build_memory_agent,
    build_summarizer,
    make_runner,
)
from .history import NUDGE_METADATA
from .models import make_model
from .tools import text_replies_enabled

log = logging.getLogger(__name__)

# A nudge is a short second invocation: one model call to act on the
# reminder, one spare in case it first returns prose.
NUDGE_MAX_LLM_CALLS = 2

# The summarizer's effort rules, enforced by the tool layer: calls per run
# and the most an examine_resource result may put into its context. Every
# extra examine is another few thousand prompt tokens on each later call,
# and the model reads well past the prompt's "0-3" without a hard stop.
SUMMARIZER_TOOL_BUDGETS = {"search_resources": 4, "examine_resource": 3}
EXAMINE_RESULT_CAP = int(os.getenv("SUMMARIZER_EXAMINE_CHARS", "12000"))


async def run_until_done(runner, *, user_id: str, session_id: str, text: str,
                         run_config: RunConfig, state_delta=None) -> None:
    """Drive one invocation to completion, treating the call cap as an end."""
    message = types.Content(role="user", parts=[types.Part(text=text)])
    try:
        async for _ in runner.run_async(user_id=user_id, session_id=session_id,
                                        new_message=message,
                                        state_delta=state_delta,
                                        run_config=run_config):
            pass
    except LlmCallsLimitExceededError as exc:
        log.warning("%s: %s", session_id, exc)


async def _run_subagent(agent, app_name: str, sub_turn: dict, request: str,
                        max_llm_calls: int, finished, nudge: str) -> None:
    """Run *agent* in a fresh session until *finished()* is true, nudging
    once if the first invocation ends without it."""
    session_service = InMemorySessionService()
    runner = make_runner(agent, session_service, app_name)
    session_id = sub_turn["session_id"]
    await session_service.create_session(app_name=app_name, user_id=agent.name,
                                         session_id=session_id)
    try:
        await run_until_done(runner, user_id=agent.name, session_id=session_id,
                             text=request,
                             run_config=RunConfig(max_llm_calls=max_llm_calls))
        if not finished():
            # Prose reaches nobody. Remind once, then cut the run rather
            # than loop.
            await run_until_done(
                runner, user_id=agent.name, session_id=session_id, text=nudge,
                run_config=RunConfig(max_llm_calls=NUDGE_MAX_LLM_CALLS,
                                     custom_metadata=dict(NUDGE_METADATA)))
    finally:
        turns.release(session_id)


def _sub_context(parent: dict, agent: str, **fields) -> dict:
    return turns.new_context(
        f"{parent.get('chat_id')}:{agent}:{uuid.uuid4().hex[:8]}", agent,
        tracer=parent.get("tracer") or NullTracer(),
        executor=parent.get("executor"),
        database=parent.get("database"),
        fields_to_remove=parent.get("fields_to_remove", []),
        **fields,
    )


# ------------------------------------------------------------- summarizer

def _summarizer_request(question: str, parent: dict) -> str:
    # The summarizer judges relevance for a conversation it doesn't see, so
    # feed it the actual current user message, not a compressed proxy of it
    # (the brief's "implicit state sharing" warning).
    parts = [f"COACH QUESTION:\n{question}"]
    user_msg = parent.get("last_user_message")
    if user_msg:
        parts.append(f"USER'S CURRENT MESSAGE (verbatim):\n{user_msg}")
    parts.append(f"ACTIVE TOPIC LIBRARY: {parent.get('database')}")
    return "\n\n".join(parts)


async def run_summarizer(question: str, parent: dict) -> dict:
    """Run one summarizer research task and return the summary struct."""
    tracer = parent.get("tracer") or NullTracer()
    model_name = (os.getenv("SUMMARIZER_MODEL") or "").strip()
    effort = (os.getenv("SUMMARIZER_REASONING_EFFORT") or "").strip() or None

    tracer.emit(SUMMARIZER_AGENT, "spawn", {
        "question": clip(question, 300),
        "model": model_name or "backend default",
    })

    # Fresh, isolated tool context. existing_resources starts empty so the
    # sub-agent's search dedup only tracks its own run; found resources are
    # merged back into the coach's context afterwards so provide_file /
    # open_course_page can deliver them by id.
    sub = _sub_context(parent, SUMMARIZER_AGENT,
                       embedding_search=parent["embedding_search"],
                       tool_budgets=dict(SUMMARIZER_TOOL_BUDGETS),
                       result_caps={"examine_resource": EXAMINE_RESULT_CAP})
    sub["state"]["summary"] = None

    await _run_subagent(
        build_summarizer(make_model(model_name, effort)), "botbase-summarizer", sub,
        _summarizer_request(question, parent), SUMMARIZER_MAX_LLM_CALLS,
        finished=lambda: sub["state"]["summary"] is not None,
        nudge=("Reminder: nothing you write as text reaches the coach. "
               "Finish now by calling return_summary with your findings "
               "so far."),
    )

    summary = sub["state"]["summary"]
    if summary is None:
        # Contract holds even when the model never called return_summary:
        # the coach always receives the same struct, flagged low-confidence.
        summary = {
            "answer": sub["last_text"] or "The summarizer did not produce an answer.",
            "key_points": [],
            "resources": [],
            "confidence": "low",
            "notes": (
                "Summarizer run ended without a structured return_summary "
                "call; treat this answer with caution."
            ),
        }

    # Provenance — two fields, per the brief's 'cheap now, expensive later'.
    summary["source"] = f"library:{parent.get('database')}"
    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
    summary["tool_calls"] = sub["tool_calls"]

    # Merge retrieved resources into the coach's context (dedup by oid) so
    # the cited ids resolve in provide_file / open_course_page.
    parent_resources = parent.setdefault("existing_resources", [])
    known_oids = {r.get("oid") for r in parent_resources}
    for r in sub["existing_resources"]:
        if r.get("oid") not in known_oids:
            parent_resources.append(r)

    tracer.emit(SUMMARIZER_AGENT, "summary", {
        "confidence": summary["confidence"],
        "answer": clip(summary["answer"], 400),
        "resources": len(summary["resources"]),
        "tool_calls": sub["tool_calls"],
    }, persist=summary)

    return summary


# ----------------------------------------------------------- memory agent

def _turn_surface(turn_messages: list[dict]) -> list[str]:
    """The conversational surface of one turn: what the user said, what
    the coach said back, plus mode switches and resource deliveries as
    events. Raw model monologue and tool traffic stay out."""
    lines = []
    for msg in turn_messages or []:
        role = msg.get("role")
        if role == "user":
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"user: {content}")
        elif role == "assistant":
            # With text replies on, the model's visible text IS what the
            # user was told (thinking is exported separately as reasoning).
            if text_replies_enabled():
                content = (msg.get("content") or "").strip()
                if content:
                    lines.append(f"coach: {content}")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or tc
                name = fn.get("name") or ""
                try:
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if name == "send_message":
                    text = (args.get("message") or "").strip()
                    if text:
                        lines.append(f"coach: {text}")
                elif name == "switch_mode":
                    lines.append(
                        f"event: switched topic mode to "
                        f"{args.get('target_mode', '?')}")
                elif name in ("provide_file", "open_course_page"):
                    lines.append(
                        f"event: delivered resource "
                        f"{args.get('resource_id', '?')} to the user")
    return lines


def _memory_request(turn_messages: list[dict], parent: dict,
                    prior_notes: list[dict]) -> str:
    parts = ["TURN TRANSCRIPT (conversational surface only):"]
    surface = _turn_surface(turn_messages)
    parts.append("\n".join(surface) if surface else "(empty turn)")

    if prior_notes:
        parts.append(
            "EXISTING NOTES for this user (do NOT duplicate):\n" + "\n".join(
                f"- [{(n.get('ts') or '')[:10]}] {n.get('rendered', '')}"
                for n in prior_notes
            ))
    else:
        parts.append("EXISTING NOTES for this user: none yet.")

    resources = parent.get("existing_resources") or []
    if resources:
        parts.append(
            "RESOURCES FROM THIS TURN (only these ids may be cited, and "
            "only if actually delivered to the user):\n" + "\n".join(
                f"- {r.get('id')} — '{r.get('title', '')}' "
                f"({r.get('source_type', '')})"
                for r in resources
            ))
    else:
        parts.append("RESOURCES FROM THIS TURN: none.")

    return "\n\n".join(parts)


async def run_memory_agent(turn_messages: list[dict], parent: dict,
                           prior_notes: list[dict] | None = None) -> list[dict]:
    """Extract template memories for one completed turn.

    Returns a list of validated {"template", "slots"} records — possibly
    empty, which is a normal outcome. A run that produces no valid
    record_memories call records nothing.
    """
    tracer = parent.get("tracer") or NullTracer()
    model_name = (os.getenv("MEMORY_MODEL") or "").strip()
    effort = (os.getenv("MEMORY_REASONING_EFFORT") or "").strip() or None

    tracer.emit(MEMORY_AGENT, "extract_start", {
        "model": model_name or "backend default",
        "prior_notes": len(prior_notes or []),
    })

    sub = _sub_context(parent, MEMORY_AGENT,
                       existing_resources=parent.get("existing_resources", []))
    sub["state"]["memories"] = None

    await _run_subagent(
        build_memory_agent(make_model(model_name, effort)), "botbase-memory", sub,
        _memory_request(turn_messages, parent, prior_notes or []),
        MEMORY_MAX_LLM_CALLS,
        finished=lambda: sub["state"]["memories"] is not None,
        nudge=("Reminder: nothing you write as text is kept. Finish now "
               "by calling record_memories — with an empty list if "
               "nothing is worth recording."),
    )

    return sub["state"]["memories"] or []
