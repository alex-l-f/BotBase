"""
Adapts the tool plugins in ``tools/`` to ADK.

Each ``tools.base.BaseTool`` becomes an ADK ``BaseTool`` whose declaration
is the plugin's existing OpenAI-style JSON schema (ADK passes it through to
LiteLLM verbatim) and whose ``run_async`` calls the plugin's ``execute``
with the turn's context dict, on the turn's executor thread.

The loop-control flags the plugins set on ``context["state"]`` map onto
ADK's own signals: ``done`` (finish_turn) becomes
``actions.skip_summarization``, which ends the invocation, and a mode
switch is mirrored into session state so the instruction provider picks
up the new prompt on the next model call.

``ask_library`` is the one tool not adapted: it spawned a summarizer on the
old loop machinery, so it is re-implemented here on top of an ADK sub-run.
"""

import asyncio

from google.adk.tools.base_tool import BaseTool
from google.genai import types

from tools import get_tool, get_tools
from tools.base import BaseTool as LegacyBaseTool

from . import turns


def _declaration(schema: dict) -> types.FunctionDeclaration:
    fn = schema["function"]
    return types.FunctionDeclaration(
        name=fn["name"],
        description=fn.get("description", ""),
        parameters_json_schema=fn.get("parameters")
        or {"type": "object", "properties": {}},
    )


_TRUNCATION_HINT = (
    "NOTE: your previous response hit the output token limit, so this "
    "call's arguments arrived empty or cut off. Respond again far more "
    "concisely: shorter text, fewer items, less deliberation."
)


def _with_truncation_hint(result):
    if isinstance(result, dict):
        return {**result, "note": _TRUNCATION_HINT}
    return f"{result} {_TRUNCATION_HINT}"


def _run_finished(state: dict) -> bool:
    """The old loops' break conditions: finish_turn sets ``done``; the
    summarizer's return_summary and the memory agent's record_memories
    stash their validated payload (an ERROR result leaves it unset so the
    model can retry)."""
    return bool(state.get("done")) or state.get("summary") is not None \
        or state.get("memories") is not None


class LegacyTool(BaseTool):
    """One ``tools/`` plugin exposed to an ADK agent."""

    def __init__(self, tool: LegacyBaseTool):
        fn = tool.schema["function"]
        super().__init__(name=fn["name"], description=fn.get("description", ""))
        self._tool = tool

    def _get_declaration(self) -> types.FunctionDeclaration:
        return _declaration(self._tool.schema)

    async def run_async(self, *, args: dict, tool_context) -> object:
        turn = turns.get(tool_context.session.id)
        turn["tool_calls"] += 1

        # Per-run call budgets (turn["tool_budgets"], name -> calls left):
        # the prompt's effort rules, enforced. A refused call costs no
        # model time, which is the point.
        budgets = turn.get("tool_budgets") or {}
        if self.name in budgets:
            if budgets[self.name] <= 0:
                return (
                    f"ERROR: the {self.name} budget for this run is used up. "
                    "Work with what you already have and finish now."
                )
            budgets[self.name] -= 1

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            turn["executor"], self._tool.execute, dict(args), turn)

        # Per-tool result caps (turn["result_caps"], name -> chars): keep
        # one oversized document from filling the agent's whole context.
        cap = (turn.get("result_caps") or {}).get(self.name)
        if cap and isinstance(result, str) and len(result) > cap:
            result = (
                result[:cap]
                + f"\n[... truncated: {len(result) - cap} more characters "
                "beyond this run's read budget]"
            )

        if turn.get("truncated"):
            result = _with_truncation_hint(result)

        if _run_finished(turn["state"]):
            # A terminal tool accepted its call — stop the loop without
            # another model call.
            tool_context.actions.skip_summarization = True

        if self.name == "switch_mode":
            status = (turn.get("chat_status") or {}).get(turn.get("chat_id")) or {}
            if status.get("mode"):
                tool_context.state["mode"] = status["mode"]

        return result


class AskLibraryTool(BaseTool):
    """Coach → summarizer delegation as an ADK sub-run.

    Keeps the plugin's schema and argument checks; the research itself is
    ``adk_app.subagents.run_summarizer`` — a fresh summarizer agent in its
    own session (context isolation), whose structured summary is returned
    as this tool's result.
    """

    def __init__(self):
        self._schema = get_tool("ask_library").schema
        fn = self._schema["function"]
        super().__init__(name=fn["name"], description=fn["description"])

    def _get_declaration(self) -> types.FunctionDeclaration:
        return _declaration(self._schema)

    async def run_async(self, *, args: dict, tool_context) -> object:
        turn = turns.get(tool_context.session.id)
        turn["tool_calls"] += 1
        turn["state"]["done"] = False

        question = (args.get("question") or "").strip()
        if not question:
            return "ERROR: Missing 'question' argument in ask_library."
        if not turn.get("database"):
            return (
                "ERROR: No topic library is active. Use switch_mode to "
                "select a topic before asking the library."
            )

        from .subagents import run_summarizer
        return await run_summarizer(question, turn)


def build_tools(toolset: str | list[str] | None) -> list[BaseTool]:
    """The ADK tool list for a toolset name (see ``tools/toolsets.py``)."""
    adk_tools: list[BaseTool] = []
    for tool in get_tools(toolset):
        name = tool.schema["function"]["name"]
        if name == "ask_library":
            adk_tools.append(AskLibraryTool())
        else:
            adk_tools.append(LegacyTool(tool))
    return adk_tools
