"""Simulation runner: drives a persona LLM against a BotBase server,
then auto-evaluates the transcript against a rubric.

Each run executes on its own daemon thread; progress is written to the
database after every message so the UI can poll it live.
"""

import threading
import time
import traceback

import db
from bot_client import BotClient
from openrouter_client import chat_completion, extract_json

END_MARKER = "[END_CONVERSATION]"

PERSONA_SYSTEM_TEMPLATE = """You are role-playing as a specific person chatting with an AI assistant. Stay fully in character for the entire conversation.

YOUR CHARACTER:
{profile_text}

RULES:
- Write ONLY the character's next chat message. No narration, no quotation marks around the whole message, no stage directions, no "As {name}, I would say...".
- Write the way this person would actually type in a chat: match their vocabulary, tone, message length, and any quirks described above.
- React naturally to what the assistant says. Ask follow-ups, push back, get confused, or go off on tangents if that fits the character.
- Do not break character or reveal that you are an AI or that this is a simulation, no matter what the assistant says.
- When the conversation has reached a natural end for this character (their goal is met, or they would realistically stop replying), end your message with the exact marker {end_marker} on its own final line. Use it only when the character is genuinely done.
"""

JUDGE_SYSTEM = """You are a strict, fair evaluator of AI assistant conversations. You will be given a conversation transcript between a simulated user and an AI assistant (the "bot"), plus a rubric. Score the BOT's performance only — the simulated user's behavior is context, not what you are grading.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{
  "criteria": [
    {"key": "<criterion key>", "score": <number within the criterion's range>, "justification": "<2-4 sentences citing specific moments in the transcript>"}
  ],
  "overall_comment": "<a short paragraph summarizing the bot's overall performance>"
}
Include every criterion from the rubric exactly once, using its key verbatim."""


# run_id -> thread, so the server can report liveness
_active_runs = {}


def start_run(run_id, config):
    """Kick off a simulation run in a background thread.

    config: {persona: {...}, rubric: {...}|None, bot_url, bot_profile,
             bot_password, max_turns}
    """
    t = threading.Thread(target=_run_simulation, args=(run_id, config), daemon=True)
    _active_runs[run_id] = t
    t.start()


def is_run_active(run_id):
    t = _active_runs.get(run_id)
    return t is not None and t.is_alive()


def _persona_messages(persona, transcript):
    """Build the persona model's message list. The persona 'is' the
    assistant from the API's perspective, so roles are flipped:
    bot messages -> user, persona messages -> assistant."""
    system = PERSONA_SYSTEM_TEMPLATE.format(
        profile_text=persona["profile_text"],
        name=persona["name"],
        end_marker=END_MARKER,
    )
    messages = [{"role": "system", "content": system}]
    if not transcript:
        messages.append({
            "role": "user",
            "content": "(You are opening the chat. Write your character's first message to the assistant.)",
        })
        return messages
    for msg in transcript:
        role = "assistant" if msg["role"] == "user" else "user"
        messages.append({"role": role, "content": msg["content"]})
    return messages


def _run_simulation(run_id, config):
    persona = config["persona"]
    transcript = []
    try:
        db.update_run(run_id, status="running")

        bot = BotClient(
            config["bot_url"],
            password=config.get("bot_password"),
            profile=config["bot_profile"],
        )
        chat_id = bot.start_chat()
        db.update_run(run_id, bot_chat_id=chat_id)

        max_turns = int(config.get("max_turns", 10))
        for _ in range(max_turns):
            user_text = chat_completion(
                persona["model"],
                _persona_messages(persona, transcript),
                temperature=float(persona.get("temperature", 0.8)),
            )
            ended = END_MARKER in user_text
            user_text = user_text.replace(END_MARKER, "").strip()

            if user_text:
                transcript.append({"role": "user", "content": user_text, "ts": time.time()})
                db.update_run(run_id, transcript=transcript)

                bot_text = bot.send(user_text)
                transcript.append({"role": "bot", "content": bot_text, "ts": time.time()})
                db.update_run(run_id, transcript=transcript, full_context=bot.full_context)

            if ended:
                break

        rubric = config.get("rubric")
        if rubric:
            db.update_run(run_id, status="evaluating")
            evaluate_run(run_id, transcript, rubric)

        db.update_run(run_id, status="completed", completed_at=time.time())
    except Exception as e:
        traceback.print_exc()
        db.update_run(
            run_id, status="failed", error=f"{type(e).__name__}: {e}",
            transcript=transcript, completed_at=time.time(),
        )
    finally:
        _active_runs.pop(run_id, None)


# ------------------------------------------------------------- evaluation

def format_transcript(transcript):
    lines = []
    for msg in transcript:
        speaker = "SIMULATED USER" if msg["role"] == "user" else "BOT"
        lines.append(f"[{speaker}]\n{msg['content']}\n")
    return "\n".join(lines) if lines else "(empty conversation)"


def _judge_user_prompt(transcript, rubric):
    crit_lines = []
    for c in rubric["criteria"]:
        crit_lines.append(
            f"- key: {c['key']}\n  title: {c['title']}\n  description: {c['description']}\n"
            f"  score range: {c.get('min', 1)} (worst) to {c.get('max', 5)} (best)"
        )
    return (
        f"RUBRIC: {rubric['name']}\n"
        f"{rubric.get('description', '')}\n\n"
        f"CRITERIA:\n" + "\n".join(crit_lines) +
        "\n\nTRANSCRIPT:\n" + format_transcript(transcript) +
        "\n\nScore the bot now. Respond with only the JSON object."
    )


def evaluate_run(run_id, transcript, rubric, judge_model=None):
    """Run the LLM judge over a transcript and store the evaluation.
    Returns the new evaluation id. Raises on hard failure."""
    judge_model = judge_model or rubric.get("judge_model")
    raw = ""
    try:
        raw = chat_completion(
            judge_model,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _judge_user_prompt(transcript, rubric)},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        parsed = extract_json(raw)

        by_key = {c["key"]: c for c in rubric["criteria"]}
        scores = []
        total_weight = 0.0
        weighted_sum = 0.0
        for item in parsed.get("criteria", []):
            crit = by_key.get(item.get("key"))
            if not crit:
                continue
            lo, hi = float(crit.get("min", 1)), float(crit.get("max", 5))
            score = max(lo, min(hi, float(item.get("score", lo))))
            weight = float(crit.get("weight", 1))
            span = (hi - lo) or 1.0
            weighted_sum += weight * (score - lo) / span
            total_weight += weight
            scores.append({
                "key": crit["key"],
                "title": crit["title"],
                "score": score,
                "min": lo,
                "max": hi,
                "weight": weight,
                "justification": item.get("justification", ""),
            })

        weighted_score = round(100.0 * weighted_sum / total_weight, 1) if total_weight else None
        return db.create_evaluation(
            run_id, rubric, judge_model, scores,
            parsed.get("overall_comment", ""), weighted_score, raw,
        )
    except Exception as e:
        db.create_evaluation(
            run_id, rubric, judge_model, [], "", None, raw,
            status="failed", error=f"{type(e).__name__}: {e}",
        )
        raise
