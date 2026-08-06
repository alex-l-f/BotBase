"""
The library summarizer sub-agent (multi-agent-paradigms-2026.md §6).

A second instance of the same LLM-in-a-loop machinery the coach runs on,
with three deliberate differences:
- read-only toolset (search_resources / examine_resource) — it contributes
  intelligence, never actions; the coach keeps all writes,
- its own fresh context window per spawn (context isolation: the big corpus
  traffic never enters the coach's context),
- a fixed structured output contract enforced by the return_summary tool,
  stamped with source + timestamp so the coach can attribute.

Run on a cheaper model by setting SUMMARIZER_MODEL in .env; defaults to the
backend's default model.
"""

import os
from datetime import datetime

from tools import get_schemas, dispatch
from prompts.summarizer import PROMPT as SUMMARIZER_PROMPT
from trace import clip, NullTracer

# Hard iteration cap — one LLM call per iteration. Effort scaling inside the
# prompt keeps typical runs well below this; the cap is the termination
# backstop (missing termination is a named MAST failure mode).
MAX_ITERATIONS = 8


def _build_request(question: str, parent_context: dict) -> str:
    # The summarizer judges relevance for a conversation it doesn't see, so
    # feed it the actual current user message, not a compressed proxy of it
    # (the brief's "implicit state sharing" warning).
    parts = [f"COACH QUESTION:\n{question}"]
    user_msg = parent_context.get("last_user_message")
    if user_msg:
        parts.append(f"USER'S CURRENT MESSAGE (verbatim):\n{user_msg}")
    parts.append(f"ACTIVE TOPIC LIBRARY: {parent_context.get('database')}")
    return "\n\n".join(parts)


def run_summarizer(question: str, parent_context: dict) -> dict:
    """Run one summarizer research task and return the summary struct."""
    tracer = parent_context.get("tracer") or NullTracer()
    llm_cls = parent_context["llm_interface_cls"]
    conv_cls = parent_context["conversation_cls"]

    model = (os.getenv("SUMMARIZER_MODEL") or "").strip()
    llm = llm_cls("", model=model) if model else llm_cls("")

    tracer.emit("summarizer", "spawn", {
        "question": clip(question, 300),
        "model": model or "backend default",
    })

    # Fresh, isolated tool context. existing_resources starts empty so the
    # sub-agent's search dedup only tracks its own run; found resources are
    # merged back into the coach's context afterwards so provide_file /
    # open_course_page can deliver them by id.
    sub_state = {
        "done": False,
        "has_responded": False,
        "needs_regeneration": False,
        "summary": None,
    }
    sub_context = {
        "state": sub_state,
        "embedding_search": parent_context["embedding_search"],
        "existing_resources": [],
        "database": parent_context["database"],
        "fields_to_remove": parent_context.get("fields_to_remove", []),
        "agent": "summarizer",
    }

    conversation = conv_cls(SUMMARIZER_PROMPT)
    conversation.add_user_message(_build_request(question, parent_context))

    schemas = get_schemas("summarizer")
    tool_calls_made = 0
    idle_iterations = 0
    last_text = ""

    for _ in range(MAX_ITERATIONS):
        response, tools = llm.get_tools_completion(conversation, schemas)
        if response and response.strip():
            last_text = response.strip()
            tracer.emit("summarizer", "llm_output", {"text": clip(last_text)})

        for tool in tools:
            tracer.emit("summarizer", "tool_call", {
                "name": tool["name"],
                "args": clip(tool["arguments"], 400),
            }, persist={"name": tool["name"], "args": tool["arguments"]})
            result = dispatch(tool["name"], tool["arguments"], sub_context)
            tracer.emit("summarizer", "tool_result", {
                "name": tool["name"],
                "result": clip(result),
            }, persist={"name": tool["name"], "result": result})
            conversation.add_tool_message(tool["id"], tool["name"], result)
            tool_calls_made += 1

        if sub_state["summary"] is not None:
            break

        if not tools:
            # Prose without a return_summary call reaches nobody. Nudge once,
            # then cut the run rather than loop.
            idle_iterations += 1
            if idle_iterations >= 2:
                break
            conversation.add_user_message(
                "Reminder: nothing you write as text reaches the coach. "
                "Finish now by calling return_summary with your findings "
                "so far."
            )

    summary = sub_state["summary"]
    if summary is None:
        # Contract holds even when the model never called return_summary:
        # the coach always receives the same struct, flagged low-confidence.
        summary = {
            "answer": last_text or "The summarizer did not produce an answer.",
            "key_points": [],
            "resources": [],
            "confidence": "low",
            "notes": (
                "Summarizer run ended without a structured return_summary "
                "call; treat this answer with caution."
            ),
        }

    # Provenance — two fields, per the brief's 'cheap now, expensive later'.
    summary["source"] = f"library:{parent_context.get('database')}"
    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
    summary["tool_calls"] = tool_calls_made

    # Merge retrieved resources into the coach's context (dedup by oid) so
    # the cited ids resolve in provide_file / open_course_page.
    parent_resources = parent_context.setdefault("existing_resources", [])
    known_oids = {r.get("oid") for r in parent_resources}
    for r in sub_context["existing_resources"]:
        if r.get("oid") not in known_oids:
            parent_resources.append(r)

    tracer.emit("summarizer", "summary", {
        "confidence": summary["confidence"],
        "answer": clip(summary["answer"], 400),
        "resources": len(summary["resources"]),
        "tool_calls": tool_calls_made,
    }, persist=summary)

    return summary
