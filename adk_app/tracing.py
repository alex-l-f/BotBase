"""
Per-agent instrumentation as an ADK plugin.

The old loops called the tracer by hand around every model and tool call.
Under ADK the same events come out of the plugin lifecycle: one plugin on
every runner streams ``llm_output`` / ``tool_call`` / ``tool_result`` to
the debug UI and the trace log through the turn's ``AgentTracer``, keyed
to the pane of whichever agent is running (coach, summarizer, memory).

It also carries the two model-quirk fixes the old adapters had: cleaning
harmony control tokens that gpt-oss leaks into function names, and
recording the full tool-call list of a response so ``finish_turn`` can
refuse to run alongside other tools.
"""

import time

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from trace import clip

from . import turns


def _visible_text(content) -> str:
    if not content or not content.parts:
        return ""
    return "\n".join(
        p.text for p in content.parts if p.text and not p.thought
    ).strip()


class TracePlugin(BasePlugin):

    def __init__(self):
        super().__init__(name="botbase_trace")

    async def before_model_callback(self, *, callback_context, llm_request):
        turn = turns.find(callback_context.session.id)
        if turn is not None:
            turn["_model_started"] = time.monotonic()
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        turn = turns.find(callback_context.session.id)
        if turn is None:
            return None

        # One llm_call event per model round trip: wall time and token
        # usage, so the trace log shows where a turn's seconds went.
        started = turn.pop("_model_started", None)
        usage = llm_response.usage_metadata
        stats = {
            "seconds": round(time.monotonic() - started, 1) if started else None,
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        }
        # A response cut off by the output cap loses its tool-call arguments
        # (LiteLLM dispatches the truncated JSON as {}). Flag it so the tool
        # result can tell the model why, instead of a bare argument error.
        turn["truncated"] = (
            llm_response.finish_reason == types.FinishReason.MAX_TOKENS)
        if turn["truncated"]:
            stats["truncated"] = True
        turn["tracer"].emit(turn["agent"], "llm_call", stats)

        if not llm_response.content:
            return None

        calls = []
        for part in llm_response.content.parts or []:
            fc = part.function_call
            if fc is None:
                continue
            # gpt-oss (harmony) tool-call parsing can leak channel control
            # tokens into the function name, e.g.
            # "search_resources<|channel|>commentary". Keep only the real
            # identifier so the call dispatches and replays cleanly.
            if fc.name and "<|" in fc.name:
                fc.name = fc.name.split("<|", 1)[0].strip()
            calls.append({"name": fc.name, "arguments": fc.args or {}})
        turn["tools_in_response"] = calls

        text = _visible_text(llm_response.content)
        if text:
            turn["last_text"] = text
            turn["tracer"].emit(turn["agent"], "llm_output", {"text": clip(text)})
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        turn = turns.find(tool_context.session.id)
        if turn is not None:
            turn["tracer"].emit(turn["agent"], "tool_call", {
                "name": tool.name,
                "args": clip(tool_args, 400),
            }, persist={"name": tool.name, "args": tool_args})
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        turn = turns.find(tool_context.session.id)
        if turn is not None:
            turn["tracer"].emit(turn["agent"], "tool_result", {
                "name": tool.name,
                "result": clip(result),
            }, persist={"name": tool.name, "result": result})
        return None
