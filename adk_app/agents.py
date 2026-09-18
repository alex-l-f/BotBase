"""
The three ADK agents and their runners.

Coach, summarizer and memory agent are ``LlmAgent``s that differ in
prompt, toolset, model and call budget; the supervisor topology of
``multi-agent-paradigms-2026.md`` §6 is preserved by running the two
helpers as sub-runs in their own sessions (``adk_app.subagents``) rather
than as ADK sub-agents, which would share the coach's session and let the
model transfer control to them.

The coach's instruction is a provider, not a string: it is recomputed on
every model call from the session's current mode, so ``switch_mode`` takes
effect mid-turn without rewriting history, and the memory snapshot is
appended each time so it survives the swap.
"""

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner

from prompts import get_prompt
from prompts.memory_agent import PROMPT as MEMORY_AGENT_PROMPT
from prompts.summarizer import PROMPT as SUMMARIZER_PROMPT
from prompts.topics import TOPICS

from . import turns
from .models import GENERATE_CONFIG
from .tools import build_tools
from .tracing import TracePlugin

COACH_AGENT = "coach"
SUMMARIZER_AGENT = "summarizer"
MEMORY_AGENT = "memory"

# Per-invocation LLM call caps: the termination backstops the old loops
# enforced with their iteration counters (one model call per iteration).
COACH_MAX_LLM_CALLS = 20
SUMMARIZER_MAX_LLM_CALLS = 8
MEMORY_MAX_LLM_CALLS = 3


def coach_instruction(ctx: ReadonlyContext) -> str:
    turn = turns.get(ctx.session.id)
    if turn.get("system_prompt"):
        return turn["system_prompt"]
    mode = ctx.state.get("mode")
    profile = mode if mode in TOPICS else turn.get("profile")
    prompt = get_prompt(profile or "default", turn.get("arch", "single"))
    return prompt + (turn.get("profile_block") or "")


def build_coach(model: str | BaseLlm, toolset) -> LlmAgent:
    return LlmAgent(
        name=COACH_AGENT,
        model=model,
        instruction=coach_instruction,
        tools=build_tools(toolset),
        generate_content_config=GENERATE_CONFIG,
    )


def build_summarizer(model: str | BaseLlm) -> LlmAgent:
    return LlmAgent(
        name=SUMMARIZER_AGENT,
        model=model,
        instruction=SUMMARIZER_PROMPT,
        tools=build_tools("summarizer"),
        generate_content_config=GENERATE_CONFIG,
    )


def build_memory_agent(model: str | BaseLlm) -> LlmAgent:
    return LlmAgent(
        name=MEMORY_AGENT,
        model=model,
        instruction=MEMORY_AGENT_PROMPT,
        tools=build_tools("memory_agent"),
        generate_content_config=GENERATE_CONFIG,
    )


def make_runner(agent: LlmAgent, session_service, app_name: str) -> Runner:
    app = App(name=app_name, root_agent=agent, plugins=[TracePlugin()])
    return Runner(app=app, session_service=session_service)
