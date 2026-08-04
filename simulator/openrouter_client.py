"""Minimal OpenRouter chat client for the simulator.

Deliberately lightweight (requests only) — no tokenizers, no OpenAI SDK —
so the simulator stays dependency-free from the main project.
"""

import json
import os
import re
import time

import requests
from dotenv import load_dotenv

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load .env from the simulator dir first, then fall back to the parent
# project's .env so a single OPENROUTER_API_KEY can serve both.
_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_here, ".env"))
load_dotenv(os.path.join(os.path.dirname(_here), ".env"))


class OpenRouterError(RuntimeError):
    pass


def _api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY not set. Put it in simulator/.env or the parent project's .env"
        )
    return key


def chat_completion(model, messages, temperature=0.8, max_tokens=1024, max_retries=4):
    """Return the assistant text for a plain chat completion."""
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    delay = 2
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                # Some OpenRouter errors come back 200 with an "error" body.
                err = data.get("error", {}).get("message", "no choices in response")
                last_err = OpenRouterError(err)
                time.sleep(delay)
                delay *= 2
                continue
            content = choices[0].get("message", {}).get("content") or ""
            return content.strip()
        except (requests.RequestException, ValueError) as e:
            last_err = OpenRouterError(str(e))
            time.sleep(delay)
            delay *= 2

    raise last_err or OpenRouterError("OpenRouter request failed")


def extract_json(text):
    """Pull the first JSON object out of an LLM response (handles ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in response")
        text = text[start:end + 1]
    return json.loads(text)
