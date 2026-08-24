import re
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime
from typing import List, Dict
from queue import Queue
import uuid
from tools import load_tools, get_schemas, dispatch
from prompts import get_prompt, get_toolset
from prompts.topics import TOPICS
from embedding_client import EmbeddingSearchClient
from memory import MemoryStore
from memory_agent import run_memory_agent
from trace import AgentTracer, clip


def _initial_database_for_profile(profile: str | None) -> str | None:
    """Pick the search-provider key the agent loop should start with.

    Topic profiles point at their per-topic index. The router has no library.
    The legacy 'default' profile keeps using the original 'imported' index
    so existing demo setups don't break.
    """
    if profile in TOPICS:
        return TOPICS[profile].get("provider")
    return "imported"

load_tools()

BACKENDS = {
    "openrouter": ("LMInterface.openrouter_interface", "OpenRouter_Interface"),
    "llama_cpp":  ("LMInterface.lcpp_interface",       "LCPP_Interface"),
}

LLMInterface = None    # populated by set_backend()
Conversation = None    # ditto
_active_backend = None


def set_backend(name: str) -> None:
    """Select the LLM backend to use for subsequent get_LM_response calls."""
    global LLMInterface, Conversation, _active_backend
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Choices: {sorted(BACKENDS)}"
        )
    module_name, class_name = BACKENDS[name]
    import importlib
    module = importlib.import_module(module_name)
    LLMInterface = getattr(module, class_name)
    Conversation = module.Conversation
    _active_backend = name


def get_active_backend() -> str | None:
    return _active_backend


# Default to OpenRouter so anything that imports this module before
# set_backend() runs still has a working interface.
set_backend("openrouter")


def clean_tool_calls(text: str) -> str:
    """Remove tool call blocks from text while preserving other content"""
    while "<function>" in text and "</function>" in text:
        start = text.find("<function>")
        end = text.find("</function>") + len("</function>")
        text = text[:start] + text[end:]

    while "<result>" in text and "</result>" in text:
        start = text.find("<result>")
        end = text.find("</result>") + len("</result>")
        text = text[:start] + text[end:]

    text = re.sub(r'<think>', '', text, flags=re.DOTALL)

    return "\n".join(line for line in text.split("\n") if line.strip())


# Global message queues and chat status tracking
message_queues = {}
chat_status = {}

# Profile-scoped template memory + instrumentation store. Single-writer:
# add_memories() is called from exactly one place (the memory agent's
# output at the end of get_LM_response); everything else only reads.
# Memory runs only in the multi architecture with an active user profile —
# the single-agent baseline stays a memory-free comparison target.
memory_store = MemoryStore()

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')


def log_conversation(chat_id: str, message_data: dict):
    """Log conversation messages to a JSON file"""
    log_file = f'logs/{chat_id}.json'

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {
            'chat_id': chat_id,
            'created_at': datetime.now().isoformat(),
            'messages': []
        }

    message_data['timestamp'] = datetime.now().isoformat()
    log_data['messages'].append(message_data)

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def create_chat_session():
    chat_id = str(uuid.uuid4())
    message_queues[chat_id] = Queue()
    chat_status[chat_id] = {"is_complete": False}
    return chat_id


def get_messages(chat_id):
    if chat_id not in message_queues:
        return []

    messages = []
    while not message_queues[chat_id].empty():
        messages.append(message_queues[chat_id].get())

    return messages


def is_chat_complete(chat_id):
    return chat_id in chat_status and chat_status[chat_id]["is_complete"]


def reset_complete(chat_id):
    if chat_id in chat_status:
        chat_status[chat_id]["is_complete"] = False


def start_conversation(system_prompt: str = None, profile: str = None, arch: str = "single"):
    if system_prompt is not None:
        prompt = system_prompt
    else:
        prompt = get_prompt(profile or "default", arch)
    conversation = Conversation(prompt)
    return conversation


def get_LM_response(conversation_dict: Dict[str, str], chat_id: str, model: str = None, system_prompt: str = None, toolset=None, profile: str = None, arch: str = "single", user_id: int | None = None):
    llm_interface = LLMInterface("") #stick to defualt for now until integrated into the frontend
    conversation = start_conversation(system_prompt, profile, arch)
    if toolset is None:
        toolset = get_toolset(profile or "default", arch)
    conversation.append_history(conversation_dict)

    last_user_message = None
    for msg in reversed(conversation_dict):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break

    state = {
        "has_responded": False,
        "has_regenerated": False,
        "needs_regeneration": False,
        "done": False,
        "response_text": "",
    }

    embedding_search = EmbeddingSearchClient()

    tracer = AgentTracer(chat_id, message_queues, memory_store)

    # The one always-injected memory tier: a small, hard-capped snapshot of
    # the active profile's structured memory notes. Multi-agent only — the
    # single-agent baseline stays a pure comparison target.
    profile_block = ""
    if arch == "multi" and user_id:
        profile_block = memory_store.profile_block(user_id,
                                                   exclude_chat=chat_id)
        if profile_block and conversation.history and \
                conversation.history[0].get("role") == "system":
            conversation.history[0]["content"] += profile_block
            tracer.emit("memory", "inject", {"chars": len(profile_block)},
                        persist={"block": profile_block})

    context = {
        "message_queues": message_queues,
        "chat_id": chat_id,
        "last_user_message": last_user_message,
        "conversation_history": conversation.history,
        "state": state,
        "embedding_search": embedding_search,
        "existing_resources": [],
        "database": _initial_database_for_profile(profile),
        "fields_to_remove": ["embedding"],
        "chat_status": chat_status,
        # Multi-agent plumbing: which architecture this turn runs, the
        # tracer + memory reader for tools, the injected snapshot (so
        # switch_mode can preserve it across a prompt swap), and the LLM
        # classes so ask_library can spawn a summarizer without importing
        # this module back.
        "arch": arch,
        "tracer": tracer,
        "memory_store": memory_store,
        "user_id": user_id,
        "profile_block": profile_block,
        "llm_interface_cls": LLMInterface,
        "conversation_cls": Conversation,
    }

    tool_schemas = get_schemas(toolset)

    tracer.emit("coach", "turn_start", {
        "arch": arch,
        "user_message": clip(last_user_message, 300),
    })

    for i in range(20):
        response, tools = llm_interface.get_tools_completion(conversation, tool_schemas)

        if response is None:
            response = ''

        if response.strip():
            tracer.emit("coach", "llm_output", {"text": clip(response)})

        # Expose this turn's full tool-call list so individual tools can
        # reason about siblings (e.g. finish_turn refuses to run when
        # other tools were called in the same model response).
        context["tools_in_response"] = tools

        for tool in tools:
            tracer.emit("coach", "tool_call", {
                "name": tool["name"],
                "args": clip(tool["arguments"], 400),
            }, persist={"name": tool["name"], "args": tool["arguments"]})
            result = dispatch(tool["name"], tool["arguments"], context)
            tracer.emit("coach", "tool_result", {
                "name": tool["name"],
                "result": clip(result),
            }, persist={"name": tool["name"], "result": result})
            conversation.add_tool_message(tool["id"], tool["name"], result)

        if state["needs_regeneration"]:
            if state["has_regenerated"]:
                if chat_id in message_queues:
                    message_queues[chat_id].put({
                        "content": "I'm sorry, but I'm unable to continue this conversation. Please refresh the page to keep chatting.",
                        "role": "assistant",
                    })
                conversation.history.pop()
                break
            else:
                state["has_regenerated"] = True
                state["needs_regeneration"] = False
                conversation.history.pop()

        if len(tools) == 0 and state["has_responded"]:
            state["done"] = True

        if state["done"]:
            break

    # Collect all the new messages from this turn
    new_messages = []
    full_context = []
    for msg in conversation.history:
        if msg["role"] != "system":
            full_context.append(msg)
            if msg not in conversation_dict:
                new_messages.append(msg)

    # Memory extraction — the single writer's only call site. The memory
    # agent turns this turn's conversational surface into anonymous
    # template records (structured generation via record_memories); only
    # what fits the template registry can be stored. Runs only in the
    # multi architecture with an active user profile.
    if arch == "multi" and user_id:
        turn_messages = []
        if last_user_message:
            turn_messages.append({"role": "user", "content": last_user_message})
        turn_messages.extend(new_messages)
        try:
            prior_notes = memory_store.list_memories(
                user_id, limit=20)["memories"]
            records = run_memory_agent(turn_messages, context, prior_notes)
            stored = memory_store.add_memories(user_id, chat_id, records) \
                if records else []
            tracer.emit("memory", "write", {
                "rows": len(stored),
                "profile": user_id,
            }, persist={"stored": stored})
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Memory extraction failed: %s", exc)

    tracer.emit("coach", "turn_done", {})

    if chat_id in chat_status:
        chat_status[chat_id]["is_complete"] = True

    return state["response_text"], new_messages, full_context
