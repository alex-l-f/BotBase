"""
The memory extraction sub-agent (privacy-preserving memory system).

Runs once at the end of each completed coach turn, on the same LLM-in-a-
loop machinery as the summarizer, with the tightest contract in the
system: its ONLY tool is record_memories, whose schema admits nothing but
registered templates with enum/reference slots (structured generation).
Whatever doesn't fit a template cannot be stored — that is the privacy
guarantee, enforced at the tool layer rather than by prompt goodwill.

The agent sees the turn's conversational surface and the profile's
existing notes (for dedup), proposes template records, and the caller
hands the validated result to MemoryStore.add_memories — the store stays
the single writer.

Run on a cheaper model by setting MEMORY_MODEL in .env; defaults to the
backend's default model.
"""

import json
import os

from tools import get_schemas, dispatch
from prompts.memory_agent import PROMPT as MEMORY_AGENT_PROMPT
from trace import clip, NullTracer

# One LLM call per iteration; 3 leaves room for one validation retry and
# one nudge. The cap is the termination backstop.
MAX_ITERATIONS = 3


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


def _build_request(turn_messages: list[dict], parent_context: dict,
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

    resources = parent_context.get("existing_resources") or []
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


def run_memory_agent(turn_messages: list[dict], parent_context: dict,
                     prior_notes: list[dict] | None = None) -> list[dict]:
    """Extract template memories for one completed turn.

    Returns a list of validated {"template", "slots"} records — possibly
    empty, which is a normal outcome. Never raises on model misbehaviour;
    a run that produces no valid record_memories call records nothing.
    """
    tracer = parent_context.get("tracer") or NullTracer()
    llm_cls = parent_context["llm_interface_cls"]
    conv_cls = parent_context["conversation_cls"]

    model = (os.getenv("MEMORY_MODEL") or "").strip()
    llm = llm_cls("", model=model) if model else llm_cls("")

    tracer.emit("memory", "extract_start", {
        "model": model or "backend default",
        "prior_notes": len(prior_notes or []),
    })

    sub_state = {"done": False, "memories": None}
    sub_context = {
        "state": sub_state,
        "existing_resources": parent_context.get("existing_resources", []),
        "database": parent_context.get("database"),
        "agent": "memory",
    }

    conversation = conv_cls(MEMORY_AGENT_PROMPT)
    conversation.add_user_message(
        _build_request(turn_messages, parent_context, prior_notes or []))

    schemas = get_schemas("memory_agent")
    nudged = False

    for _ in range(MAX_ITERATIONS):
        response, tools = llm.get_tools_completion(conversation, schemas)
        if response and response.strip():
            tracer.emit("memory", "llm_output",
                        {"text": clip(response.strip())})

        for tool in tools:
            tracer.emit("memory", "tool_call", {
                "name": tool["name"],
                "args": clip(tool["arguments"], 400),
            }, persist={"name": tool["name"], "args": tool["arguments"]})
            result = dispatch(tool["name"], tool["arguments"], sub_context)
            tracer.emit("memory", "tool_result", {
                "name": tool["name"],
                "result": clip(result),
            }, persist={"name": tool["name"], "result": result})
            conversation.add_tool_message(tool["id"], tool["name"], result)

        if sub_state["memories"] is not None:
            break

        if not tools:
            # Prose records nothing. Nudge once, then cut the run.
            if nudged:
                break
            nudged = True
            conversation.add_user_message(
                "Reminder: nothing you write as text is kept. Finish now "
                "by calling record_memories — with an empty list if "
                "nothing is worth recording."
            )

    return sub_state["memories"] or []
