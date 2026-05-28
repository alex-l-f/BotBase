from .base import BaseTool


def _lookup_resource(resource_id: str, context: dict) -> dict | None:
    """
    Find a resource the bot already pulled in this turn (most common path),
    or fall back to a direct DB lookup. Mirrors the behaviour of
    examine_resource so the bot doesn't have to call examine first.
    """
    for r in context.get("existing_resources", []):
        if r.get("id") == resource_id:
            return r

    if not resource_id.startswith("RES_"):
        return None
    try:
        oid = int(resource_id[4:])
    except ValueError:
        return None

    database = context.get("database")
    if not database:
        return None

    embedding_search = context["embedding_search"]
    embedding_search.switch_provider(database)
    details = embedding_search.get_resource_details(oid)
    if not details:
        return None

    for field in context.get("fields_to_remove", []):
        details.pop(field, None)
    details["portalURL"] = details.get("portal_url", "")
    details["oid"] = details["id"]
    details["id"] = f"RES_{str(details['id']).zfill(5)}"
    context.setdefault("existing_resources", []).append(details)
    return details


class ProvideFile(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "provide_file",
            "description": (
                "Send a specific resource file directly to the user. The "
                "frontend will embed a video/audio player or render a "
                "download link as appropriate for the file type. Use this "
                "when the user has agreed they want the file, or when "
                "the most helpful response is to hand them the source "
                "material (a guided practice audio, walkthrough video, "
                "or skill-summary PDF). Always send a brief send_message "
                "FIRST introducing the file, then call provide_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {
                        "type": "string",
                        "description": (
                            "Exact resource ID, e.g. 'RES_00012'. Get this "
                            "from a previous search_resources call."
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "Optional 1-line note shown alongside the file "
                            "(e.g. 'A 10-minute guided breathing practice')."
                        ),
                    },
                },
                "required": ["resource_id"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        state = context["state"]
        state["done"] = False

        resource_id = arguments.get("resource_id")
        if not resource_id:
            return "ERROR: Missing 'resource_id' argument."

        resource = _lookup_resource(resource_id, context)
        if resource is None:
            return (
                f"ERROR: Could not find resource {resource_id}. "
                "Search first with search_resources."
            )

        url = resource.get("portalURL") or resource.get("portal_url") or ""
        if not url:
            return f"ERROR: Resource {resource_id} has no portal_url."

        chat_id = context.get("chat_id")
        message_queues = context.get("message_queues") or {}
        note = arguments.get("note", "") or ""

        payload = {
            "role": "assistant_file",
            "resource_id": resource_id,
            "title": resource.get("title", ""),
            "source_type": resource.get("source_type") or "",
            "source_file": resource.get("source_file") or "",
            "url": url,
            "note": note,
            "summary": resource.get("description", "")[:240],
        }

        if chat_id in message_queues:
            message_queues[chat_id].put(payload)

        # Mark that the bot has produced user-facing output this turn so
        # finish_turn is willing to end the turn.
        state["has_responded"] = True

        return (
            f"Sent file '{resource.get('title', resource_id)}' "
            f"({payload['source_type']}) to user. URL: {url}"
        )
