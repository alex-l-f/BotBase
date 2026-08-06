"""
Smoke tests for the multi-agent architecture (coach / summarizer / memory).

Everything runs against fakes — no LLM backend, no embedding service, no
API keys. Covers the contracts the brief says are cheap now and expensive
later: the summarizer's structured output, single-writer memory, provenance
fields, and the arch-aware prompt/toolset resolution.
"""

import json

import pytest

from memory import MemoryStore
from tools import load_tools, get_schemas, dispatch

load_tools()


# --------------------------------------------------------------------- fakes

class FakeEmbeddingSearch:
    """Stands in for EmbeddingSearchClient."""

    RESOURCES = {
        1: {
            "id": 1,
            "title": "Stress Fact Sheet",
            "description": "What stress is and how the body responds. " * 5,
            "physical_address": "",
            "portal_url": "api/file/coping_stress/1",
            "source_type": "pdf",
            "embedding": b"blob",
        },
    }

    def switch_provider(self, provider):
        self.provider = provider

    def search(self, queries, language="all", k=10):
        return {1: 0.91}

    def get_resource_details(self, resource_id):
        details = self.RESOURCES.get(resource_id)
        return dict(details) if details else None


class FakeConversation:
    def __init__(self, system_prompt=None):
        self.history = []
        if system_prompt is not None:
            self.history = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, message):
        self.history.append({"role": "user", "content": message})

    def add_tool_message(self, tool_call_id, tool_name, tool_response):
        if isinstance(tool_response, dict):
            tool_response = json.dumps(tool_response)
        self.history.append({
            "role": "tool", "tool_call_id": tool_call_id,
            "name": tool_name, "content": tool_response,
        })


def make_fake_llm_cls(script):
    """An LLM interface class whose get_tools_completion pops from *script*:
    a list of (text, tool_calls) tuples."""

    class FakeLLM:
        def __init__(self, system_prompt, model=None, **kwargs):
            self.model = model

        def get_tools_completion(self, conversation, tools,
                                 max_tokens=2048, stop_sequences=None):
            if script:
                return script.pop(0)
            return "", []

    return FakeLLM


def make_parent_context(llm_cls, tmp_path):
    return {
        "state": {"done": False, "has_responded": False},
        "embedding_search": FakeEmbeddingSearch(),
        "existing_resources": [],
        "database": "coping_stress",
        "fields_to_remove": ["embedding"],
        "last_user_message": "why does my heart race when I'm stressed?",
        "llm_interface_cls": llm_cls,
        "conversation_cls": FakeConversation,
        "memory_store": MemoryStore(str(tmp_path / "mem.db")),
        "chat_id": "chat-current",
    }


# -------------------------------------------------------------------- memory

def _sample_turn_messages():
    return [
        {"role": "user", "content": "I want a breathing exercise"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {
                    "name": "send_message",
                    "arguments": json.dumps(
                        {"message": "Let's try tactical breathing."}),
                }},
                {"function": {
                    "name": "switch_mode",
                    "arguments": json.dumps(
                        {"target_mode": "coping_mental_skills",
                         "reason": "user wants a technique"}),
                }},
            ],
        },
        {"role": "tool", "name": "send_message", "content": "Queued."},
    ]


def test_memory_record_and_search(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    rows = store.record_turn("chat-a", "coping_mental_skills",
                             _sample_turn_messages())
    assert rows == 3  # user + assistant + mode-switch event

    hits = store.search("breathing", exclude_chat="chat-b")
    assert hits, "expected a hit for 'breathing'"
    assert hits[0]["source"] == "memory.db episodic log"
    assert hits[0]["when"]  # timestamp present

    # Excluding the chat that owns the rows hides them.
    assert store.search("breathing", exclude_chat="chat-a") == []


def test_memory_profile_block(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    assert store.profile_block() == ""  # empty store → no injection

    store.record_turn("chat-a", "coping_mental_skills",
                      _sample_turn_messages())
    block = store.profile_block(exclude_chat="chat-b", max_chars=1200)
    assert "Prior sessions with this user: 1" in block
    assert "breathing" in block  # recent user ask surfaced
    assert len(block) <= 1240  # hard cap (+ truncation marker slack)

    # The chat's own rows don't count toward its snapshot.
    assert store.profile_block(exclude_chat="chat-a") == ""


def test_memory_search_tool(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    store.record_turn("chat-old", "performance", _sample_turn_messages())
    context = {
        "state": {"done": True},
        "memory_store": store,
        "chat_id": "chat-current",
    }
    result = dispatch("memory_search", {"query": "breathing"}, context)
    assert isinstance(result, dict) and result["hits"]
    assert context["state"]["done"] is False

    result = dispatch("memory_search", {"query": "zzz-nothing"}, context)
    assert "normal outcome" in result


# ------------------------------------------------------- prompts & toolsets

def test_arch_aware_prompt_and_toolset():
    from prompts import get_prompt, get_toolset

    single = get_prompt("coping_stress", arch="single")
    multi = get_prompt("coping_stress", arch="multi")
    assert "MULTI-AGENT MODE" not in single
    assert "MULTI-AGENT MODE" in multi
    assert multi.startswith(single)  # overlay appends, never rewrites

    assert get_toolset("coping_stress", arch="single") == "topic_bot"
    assert get_toolset("coping_stress", arch="multi") == "coach"
    assert get_toolset("default", arch="multi") == "default"

    coach_tools = {s["function"]["name"] for s in get_schemas("coach")}
    assert "ask_library" in coach_tools and "memory_search" in coach_tools
    assert "search_resources" not in coach_tools  # research is delegated

    summarizer_tools = {s["function"]["name"] for s in get_schemas("summarizer")}
    assert summarizer_tools == {"search_resources", "examine_resource",
                                "return_summary"}


# --------------------------------------------------------------- summarizer

def test_return_summary_rejects_invented_ids():
    context = {"state": {}, "existing_resources": [{"id": "RES_00001"}]}
    result = dispatch("return_summary", {
        "answer": "An answer.",
        "confidence": "high",
        "resources": [{"id": "RES_09999"}],
    }, context)
    assert result.startswith("ERROR")
    assert context["state"].get("summary") is None


def test_run_summarizer_contract_and_resource_merge(tmp_path):
    script = [
        ("", [{"id": "t1", "name": "search_resources",
               "arguments": {"query": "stress physiology"}}]),
        ("", [{"id": "t2", "name": "return_summary", "arguments": {
            "answer": "Stress triggers the fight-or-flight response.",
            "key_points": ["heart rate rises"],
            "resources": [{"id": "RES_00001", "title": "Stress Fact Sheet",
                           "source_type": "pdf",
                           "why_relevant": "explains the physiology"}],
            "confidence": "high",
        }}]),
    ]
    parent = make_parent_context(make_fake_llm_cls(script), tmp_path)

    from summarizer_agent import run_summarizer
    summary = run_summarizer("What explains the physiology of stress?", parent)

    # Fixed output contract, provenance included.
    for field in ("answer", "key_points", "resources", "confidence",
                  "source", "timestamp", "tool_calls"):
        assert field in summary, f"missing contract field {field}"
    assert summary["confidence"] == "high"
    assert summary["source"] == "library:coping_stress"
    assert summary["resources"][0]["id"] == "RES_00001"
    assert summary["tool_calls"] == 2

    # Retrieved resources merged back so provide_file can resolve the id.
    assert any(r.get("id") == "RES_00001"
               for r in parent["existing_resources"])


def test_run_summarizer_fallback_without_structured_return(tmp_path):
    script = [("Here is some prose that never becomes a summary.", [])]
    parent = make_parent_context(make_fake_llm_cls(script), tmp_path)

    from summarizer_agent import run_summarizer
    summary = run_summarizer("Anything?", parent)

    assert summary["confidence"] == "low"
    assert "prose" in summary["answer"]
    assert summary["resources"] == []


def test_ask_library_requires_active_library():
    context = {"state": {"done": True}, "database": None}
    result = dispatch("ask_library", {"question": "anything"}, context)
    assert result.startswith("ERROR")
    assert "switch_mode" in result
