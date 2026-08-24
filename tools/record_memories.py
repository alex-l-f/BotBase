import json

from .base import BaseTool
from memory_templates import (
    TEMPLATES,
    registry_prompt_lines,
    slot_union_properties,
    validate_memory,
)


class RecordMemories(BaseTool):
    """The memory agent's fixed output contract (structured generation).

    Only exposed in the 'memory_agent' toolset. Ends the extraction run by
    stashing a list of validated template records in its state; the caller
    (memory_agent.run_memory_agent) hands them to MemoryStore.add_memories,
    which stays the single writer.

    This schema is the privacy enforcement point: every slot is an enum or
    a resource id retrieved this session, so nothing the user literally
    said can be stored. Invalid entries are rejected with a description of
    the problem so the model can correct and retry.
    """

    schema = {
        "type": "function",
        "function": {
            "name": "record_memories",
            "description": (
                "Record the durable, anonymous memory notes from this "
                "session. REQUIRED final step of every run — nothing you "
                "write outside this call is kept. Pass an empty list when "
                "nothing new is worth remembering (a normal outcome).\n\n"
                "Available templates:\n" + "\n".join(registry_prompt_lines())
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "description": (
                            "New memory notes, one per template instance. "
                            "Do not repeat notes that already exist."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "template": {
                                    "type": "string",
                                    "enum": list(TEMPLATES.keys()),
                                },
                                **slot_union_properties(),
                            },
                            "required": ["template"],
                        },
                    },
                },
                "required": ["memories"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        entries = arguments.get("memories")
        if not isinstance(entries, list):
            return "ERROR: record_memories requires a 'memories' array."

        # Resource ids the parent turn actually retrieved: RES_xxxxx → row.
        known_resources = {
            r.get("id"): r for r in context.get("existing_resources", [])
        }
        provider = context.get("database")

        records = []
        problems = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                problems.append(f"entry {i}: not an object")
                continue
            template = entry.get("template")
            slots = {k: v for k, v in entry.items() if k != "template"}

            # Resolve a RES_xxxxx reference against this session's results.
            if slots.get("resource") is not None:
                res = known_resources.get(str(slots["resource"]))
                if res is None:
                    problems.append(
                        f"entry {i} ({template}): resource "
                        f"{slots['resource']!r} was not retrieved this "
                        "session — cite only ids from the session summary")
                    continue
                slots["resource"] = res.get("oid")
                slots["provider"] = provider
                slots["title"] = res.get("title", "")

            clean, errors = validate_memory(template, slots)
            if errors:
                problems.append(f"entry {i} ({template}): " + "; ".join(errors))
                continue
            records.append({"template": template, "slots": clean})

        if problems:
            return (
                "ERROR: some entries were rejected — fix them and call "
                "record_memories again with the full corrected list:\n"
                + json.dumps(problems, ensure_ascii=False)
            )

        context["state"]["memories"] = records
        return f"Recorded {len(records)} memory note(s). Run complete."
