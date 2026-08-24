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

def _sample_records():
    return [
        {"template": "practice_commitment",
         "slots": {"skill": "tactical_breathing", "timeframe": "this_week"}},
        {"template": "stress_reported",
         "slots": {"area": "work", "level": "high"}},
    ]


def test_memory_templates_reject_free_text():
    from memory_templates import validate_memory

    # Unknown template.
    _, errors = validate_memory("diary_entry", {"text": "user said hi"})
    assert errors

    # Unregistered slot (the free-text smuggling path).
    _, errors = validate_memory(
        "stress_reported", {"area": "work", "details": "boss yelled at them"})
    assert any("free-form" in e for e in errors)

    # Off-enum value.
    _, errors = validate_memory("stress_reported", {"area": "my boss Carl"})
    assert errors

    # Valid record passes and drops nothing it shouldn't.
    clean, errors = validate_memory(
        "stress_reported", {"area": "work", "level": "high"})
    assert not errors and clean == {"area": "work", "level": "high"}


def test_memory_profiles_and_search(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    alice = store.create_profile("Alice")["id"]
    bob = store.create_profile("Bob")["id"]

    stored = store.add_memories(alice, "chat-a", _sample_records())
    assert len(stored) == 2
    assert "tactical breathing" in stored[0]["rendered"]

    # Search is profile-scoped: Alice hits, Bob doesn't.
    hits = store.search(alice, "breathing", exclude_chat="chat-b")
    assert hits and hits[0]["when"]
    assert "transcript" not in hits[0]["note"]
    assert store.search(bob, "breathing", exclude_chat="chat-b") == []

    # Excluding the chat that owns the rows hides them.
    assert store.search(alice, "breathing", exclude_chat="chat-a") == []

    # Free text can't be stored, even through the write API.
    with pytest.raises(ValueError):
        store.add_memories(alice, "chat-a", [
            {"template": "stress_reported",
             "slots": {"area": "work", "note": "verbatim user quote"}}])


def test_memory_profile_block(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    assert store.profile_block(None) == ""      # no profile → no injection

    pid = store.create_profile("Alice")["id"]
    assert store.profile_block(pid) == ""       # no memories yet

    store.add_memories(pid, "chat-a", _sample_records())
    block = store.profile_block(pid, exclude_chat="chat-b", max_chars=1200)
    assert "Alice" in block
    assert "Open follow-ups" in block            # practice_commitment
    assert "tactical breathing" in block
    assert len(block) <= 1240  # hard cap (+ truncation marker slack)

    # The chat's own rows don't count toward its snapshot.
    assert store.profile_block(pid, exclude_chat="chat-a") == ""


def test_memory_search_tool(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    pid = store.create_profile("Alice")["id"]
    store.add_memories(pid, "chat-old", _sample_records())
    context = {
        "state": {"done": True},
        "memory_store": store,
        "chat_id": "chat-current",
        "user_id": pid,
    }
    result = dispatch("memory_search", {"query": "breathing"}, context)
    assert isinstance(result, dict) and result["hits"]
    assert context["state"]["done"] is False

    result = dispatch("memory_search", {"query": "zzz-nothing"}, context)
    assert "normal outcome" in result

    # Without an active profile there is nothing to search.
    context.pop("user_id")
    result = dispatch("memory_search", {"query": "breathing"}, context)
    assert "No user profile" in result


def test_record_memories_tool_validation():
    context = {
        "state": {},
        "database": "coping_recovery",
        "existing_resources": [
            {"id": "RES_00004", "oid": 4, "title": "OLBI_EN",
             "source_type": "docx"},
        ],
    }
    # Invented resource id → rejected, nothing stashed.
    result = dispatch("record_memories", {"memories": [
        {"template": "resource_shared", "resource": "RES_09999"},
    ]}, context)
    assert result.startswith("ERROR")
    assert context["state"].get("memories") is None

    # Valid batch: resource resolved to provider + oid + title snapshot.
    result = dispatch("record_memories", {"memories": [
        {"template": "resource_shared", "resource": "RES_00004",
         "purpose": "self_assessment"},
        {"template": "distress_supported", "indicator": "burnout_signs",
         "action": "shared_coping_resources"},
    ]}, context)
    assert "2 memory note(s)" in result
    stored = context["state"]["memories"]
    assert stored[0]["slots"] == {
        "provider": "coping_recovery", "resource": 4, "title": "OLBI_EN",
        "purpose": "self_assessment"}

    # An empty list is a normal outcome.
    context["state"] = {}
    result = dispatch("record_memories", {"memories": []}, context)
    assert context["state"]["memories"] == []


def test_run_memory_agent_structured_extraction(tmp_path):
    script = [
        ("", [{"id": "t1", "name": "record_memories", "arguments": {
            "memories": [
                {"template": "practice_commitment",
                 "skill": "tactical_breathing", "timeframe": "this_week"},
            ]}}]),
    ]
    parent = make_parent_context(make_fake_llm_cls(script), tmp_path)

    from memory_agent import run_memory_agent
    records = run_memory_agent(
        [{"role": "user", "content": "I'll practise the breathing"}],
        parent, prior_notes=[])
    assert records == [{"template": "practice_commitment",
                        "slots": {"skill": "tactical_breathing",
                                  "timeframe": "this_week"}}]

    # Records flow into the store through the single writer.
    store = parent["memory_store"]
    pid = store.create_profile("Alice")["id"]
    stored = store.add_memories(pid, "chat-current", records)
    assert "Follow up" in stored[0]["rendered"]


def test_run_memory_agent_gives_up_gracefully(tmp_path):
    # A model that only produces prose records nothing (after one nudge).
    script = [("Interesting session!", []), ("Nothing to record.", [])]
    parent = make_parent_context(make_fake_llm_cls(script), tmp_path)

    from memory_agent import run_memory_agent
    assert run_memory_agent([], parent, prior_notes=[]) == []


def test_scenario_roundtrip_and_retime(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    pid = store.create_profile("Alice")["id"]
    store.add_memories(pid, "chat-a", _sample_records(),
                       ts="2026-01-01T10:00:00")

    exported = store.export_memories(pid)
    assert len(exported) == 2 and exported[0]["template"]

    import scenarios as scenario_mod
    retimed = scenario_mod.retime_memories(exported, newest_days_ago=1)
    assert retimed[0]["ts"] != exported[0]["ts"]

    other = store.create_profile("Bob")["id"]
    assert store.replace_memories(other, retimed) == 2
    assert store.list_memories(other)["total"] == 2

    # A malformed scenario record can't half-load.
    bad = retimed + [{"template": "diary_entry", "slots": {"text": "hi"}}]
    with pytest.raises(ValueError):
        store.replace_memories(other, bad)
    assert store.list_memories(other)["total"] == 2


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
