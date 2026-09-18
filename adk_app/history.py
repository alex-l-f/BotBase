"""
Conversion between ADK session events and the OpenAI-style message list
the frontend and simulator exchange with the server (``fullContext``).

ADK sessions are the source of truth for a chat's history; the message
list is a view of them. The reverse direction seeds a session from a
client-supplied history when the server has none for that chat (a restart,
or a client that kept its own transcript).
"""

import json

from google.adk.events import Event
from google.genai import types

# Marks the user-role reminders the runtime injects when an agent ends a
# run without calling its terminal tool. They are real session events (the
# model must see them) but not part of the conversation shown to the user.
NUDGE_METADATA = {"botbase_nudge": True}


def _is_nudge(event: Event) -> bool:
    """The reminder itself; the agent's reaction to it stays visible."""
    return event.author == "user" and bool(
        (event.custom_metadata or {}).get("botbase_nudge"))


def _response_content(response) -> str:
    # ADK wraps non-dict tool results as {"result": value}; unwrap so the
    # transcript shows what the tool actually returned.
    if isinstance(response, dict) and set(response) == {"result"}:
        response = response["result"]
    if isinstance(response, str):
        return response
    return json.dumps(response, ensure_ascii=False, default=str)


def events_to_messages(events: list[Event], include_user: bool = True) -> list[dict]:
    """Render session events as user / assistant / tool messages."""
    messages: list[dict] = []
    for event in events:
        if _is_nudge(event) or not event.content or not event.content.parts:
            continue
        parts = event.content.parts

        if event.author == "user":
            if include_user:
                text = "".join(p.text for p in parts if p.text)
                messages.append({"role": "user", "content": text})
            continue

        responses = [p.function_response for p in parts if p.function_response]
        if responses:
            for fr in responses:
                messages.append({
                    "role": "tool",
                    "tool_call_id": fr.id or "",
                    "name": fr.name or "",
                    "content": _response_content(fr.response),
                })
            continue

        message: dict = {"role": "assistant", "content": None}
        text = "".join(p.text for p in parts if p.text and not p.thought)
        reasoning = "".join(p.text for p in parts if p.text and p.thought)
        if text:
            message["content"] = text
        if reasoning:
            message["reasoning"] = reasoning
        calls = [p.function_call for p in parts if p.function_call]
        if calls:
            message["tool_calls"] = [{
                "id": fc.id or "",
                "type": "function",
                "function": {
                    "name": fc.name or "",
                    "arguments": json.dumps(fc.args or {}, ensure_ascii=False),
                },
            } for fc in calls]
        if message["content"] is not None or calls or reasoning:
            messages.append(message)
    return messages


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_result(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"result": raw}


def messages_to_events(messages: list[dict], agent_name: str) -> list[Event]:
    """Turn a client history into events that replay as ADK conversation."""
    events: list[Event] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue  # the instruction provider owns the system prompt
        if role == "user":
            text = msg.get("content") or ""
            if text:
                events.append(Event(
                    author="user", invocation_id="import",
                    content=types.Content(role="user",
                                          parts=[types.Part(text=text)])))
        elif role == "assistant":
            parts = []
            if msg.get("content"):
                parts.append(types.Part(text=msg["content"]))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or tc
                parts.append(types.Part(function_call=types.FunctionCall(
                    id=tc.get("id") or "",
                    name=fn.get("name") or "",
                    args=_parse_args(fn.get("arguments")))))
            if parts:
                events.append(Event(
                    author=agent_name, invocation_id="import",
                    content=types.Content(role="model", parts=parts)))
        elif role == "tool":
            events.append(Event(
                author=agent_name, invocation_id="import",
                content=types.Content(role="user", parts=[types.Part(
                    function_response=types.FunctionResponse(
                        id=msg.get("tool_call_id") or "",
                        name=msg.get("name") or "",
                        response=_parse_result(msg.get("content"))))])))
    return events


async def seed_session(session_service, session, messages: list[dict],
                       agent_name: str) -> None:
    for event in messages_to_events(messages, agent_name):
        await session_service.append_event(session=session, event=event)
