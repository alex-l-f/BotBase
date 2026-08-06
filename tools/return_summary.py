import json

from .base import BaseTool

_CONFIDENCE_LEVELS = ("high", "medium", "low")


class ReturnSummary(BaseTool):
    """The summarizer sub-agent's fixed output contract.

    Only exposed in the 'summarizer' toolset. Ends the summarizer run by
    stashing a validated struct in its state; the coach receives the struct
    as the ask_library tool result. Cited resource ids are checked against
    what the run actually retrieved, so an invented id is rejected instead
    of flowing to the coach (the cheapest verification available).
    """

    schema = {
        "type": "function",
        "function": {
            "name": "return_summary",
            "description": (
                "Finish the research run and hand your findings back to the "
                "coach. REQUIRED final step of every run — nothing you write "
                "outside this call reaches the coach. Cite only resource ids "
                "you actually retrieved in this run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "2-6 sentences directly answering the coach's "
                            "question. Quote canonical program definitions "
                            "verbatim where they matter."
                        ),
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Up to 6 short bullets the coach can use directly.",
                    },
                    "resources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "source_type": {"type": "string"},
                                "why_relevant": {"type": "string"},
                            },
                            "required": ["id"],
                        },
                        "description": (
                            "Resources worth delivering to the user, by exact "
                            "id from your own search results (e.g. "
                            "'RES_00012'). Never invent an id."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": list(_CONFIDENCE_LEVELS),
                        "description": (
                            "high: the library clearly answers this. medium: "
                            "partial coverage. low: the library doesn't "
                            "really cover it."
                        ),
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Gaps, contradictions, or 'library does not "
                            "cover X'. Honest emptiness beats padding."
                        ),
                    },
                },
                "required": ["answer", "confidence"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        answer = (arguments.get("answer") or "").strip()
        if not answer:
            return "ERROR: return_summary requires a non-empty 'answer'."

        confidence = (arguments.get("confidence") or "").lower()
        if confidence not in _CONFIDENCE_LEVELS:
            confidence = "low"

        known_ids = {
            r.get("id") for r in context.get("existing_resources", [])
        }
        resources = []
        unknown = []
        for r in arguments.get("resources") or []:
            if not isinstance(r, dict):
                continue
            rid = r.get("id")
            if rid in known_ids:
                resources.append({
                    "id": rid,
                    "title": r.get("title", ""),
                    "source_type": r.get("source_type", ""),
                    "why_relevant": r.get("why_relevant", ""),
                })
            else:
                unknown.append(rid)

        if unknown:
            return (
                "ERROR: return_summary cites resource ids that were not "
                f"retrieved in this run: {json.dumps(unknown)}. Cite only ids "
                "from your own search_resources results, then call "
                "return_summary again."
            )

        key_points = [
            str(p).strip() for p in (arguments.get("key_points") or [])
            if str(p).strip()
        ][:6]

        context["state"]["summary"] = {
            "answer": answer,
            "key_points": key_points,
            "resources": resources,
            "confidence": confidence,
            "notes": (arguments.get("notes") or "").strip(),
        }
        return "Summary accepted. Run complete."
