"""
LLM backends for the ADK port.

The old ``LMInterface`` adapters are replaced by ADK's ``LiteLlm`` wrapper,
which speaks to any OpenAI-compatible endpoint through LiteLLM, plus ADK's
native Gemini path. A backend is a recipe for building the model object an
``LlmAgent`` takes; ``set_backend`` picks it for the process (the server's
``--backend`` flag), ``make_model`` builds one, optionally for a different
model id (the per-agent ``SUMMARIZER_MODEL`` / ``MEMORY_MODEL`` overrides).
"""

import os

import requests
from dotenv import load_dotenv
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

# Makes OPENROUTER_API_KEY / GOOGLE_API_KEY visible to LiteLLM and google-genai,
# which read them from the environment.
load_dotenv()

# Sampling settings the previous adapters hard-coded, except the output cap:
# thinking tokens count against it, and a structured tool call that gets
# cut off is lost (LiteLLM dispatches the truncated JSON as empty args),
# costing a full round trip. 2048 was hit routinely by return_summary.
GENERATE_CONFIG = types.GenerateContentConfig(
    temperature=0.6,
    top_p=0.95,
    max_output_tokens=int(os.getenv("BOTBASE_MAX_OUTPUT_TOKENS", "8192")),
)

BACKENDS: dict[str, dict] = {
    "openrouter": {
        "help": "OpenRouter's API (needs OPENROUTER_API_KEY in .env)",
        "default_model": os.getenv("BOTBASE_MODEL", "openai/gpt-oss-20b"),
    },
    "llama_cpp": {
        "help": "a local llama.cpp server's OpenAI-compatible endpoint "
                "(LLAMA_CPP_URL, default http://localhost:8080/v1; the "
                "served model is discovered unless LLAMA_CPP_MODEL is set)",
        "default_model": os.getenv("LLAMA_CPP_MODEL", ""),
    },
    "gemini": {
        "help": "Gemini through ADK's native google-genai client "
                "(needs GOOGLE_API_KEY)",
        "default_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    },
}

# LiteLLM retries rate limits / timeouts / transient 5xx with exponential
# backoff; without this one 429 from the host fails the whole turn.
LITELLM_RETRIES = int(os.getenv("BOTBASE_LLM_RETRIES", "4"))

_active_backend = "openrouter"


def set_backend(name: str) -> None:
    """Select the backend every subsequently built agent uses."""
    global _active_backend
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Choices: {sorted(BACKENDS)}"
        )
    _active_backend = name


def get_active_backend() -> str:
    return _active_backend


def _served_model(api_base: str) -> str:
    """The model id a llama.cpp-style server is serving (it serves one).

    Some servers validate the id, so a placeholder is not enough; falls back
    to "local" when the server can't be asked.
    """
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/models", timeout=3)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return "local"


def make_model(model: str | None = None,
               reasoning_effort: str | None = None) -> str | BaseLlm:
    """Build the model object for an ``LlmAgent`` on the active backend.

    *model* overrides the backend's default model id; *reasoning_effort*
    overrides ``BOTBASE_REASONING_EFFORT`` for this one agent. Returns a
    ``LiteLlm`` for the OpenAI-compatible backends and a plain model name
    for Gemini (ADK resolves those natively).
    """
    backend = _active_backend
    model = (model or "").strip() or BACKENDS[backend]["default_model"]

    if backend == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        # OpenRouter's unified reasoning control; 'off' sends none.
        effort = (reasoning_effort or os.getenv("BOTBASE_REASONING_EFFORT")
                  or "medium").strip().lower()
        extra = {"reasoning": {"effort": effort}} if effort not in ("", "off") else {}
        return LiteLlm(model=f"openrouter/{model}", num_retries=LITELLM_RETRIES,
                       **({"extra_body": extra} if extra else {}))

    if backend == "llama_cpp":
        api_base = os.getenv("LLAMA_CPP_URL", "http://localhost:8080/v1")
        return LiteLlm(
            model=f"openai/{model or _served_model(api_base)}",
            api_base=api_base,
            api_key="-",
            num_retries=LITELLM_RETRIES,
            # llama.cpp honours this to toggle the model's thinking mode.
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

    return model
