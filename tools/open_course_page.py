from .base import BaseTool
from .provide_file import _lookup_resource


class OpenCoursePage(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "open_course_page",
            "description": (
                "Open a page from the e-learning course in the user's "
                "content viewer, right beside the chat. Use this for "
                "resources with source_type 'course_page' — interactive "
                "lessons from the online course (text, videos, practice "
                "activities). Use it when a course page is the best way to "
                "teach or practise what the user is asking about, or when "
                "they ask to see the course material. Always send a brief "
                "send_message FIRST introducing the page, then call "
                "open_course_page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {
                        "type": "string",
                        "description": (
                            "Exact resource ID, e.g. 'RES_00042'. Get this "
                            "from a previous search_resources call. The "
                            "resource must be a course page."
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "Optional 1-line note shown alongside the page "
                            "(e.g. 'A short practice activity for "
                            "identifying your stressors')."
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

        if (resource.get("source_type") or "") != "course_page":
            return (
                f"ERROR: Resource {resource_id} is a "
                f"'{resource.get('source_type') or 'file'}' resource, not a "
                "course page. Use provide_file to send files."
            )

        url = resource.get("portalURL") or resource.get("portal_url") or ""
        if not url:
            return f"ERROR: Course page {resource_id} has no URL."

        chat_id = context.get("chat_id")
        message_queues = context.get("message_queues") or {}
        note = arguments.get("note", "") or ""

        payload = {
            "role": "assistant_lesson",
            "resource_id": resource_id,
            "lesson_id": resource.get("lesson_id") or "",
            "title": resource.get("title", ""),
            "section": resource.get("physical_address") or "",
            "url": url,
            "note": note,
            "summary": resource.get("description", "")[:240],
        }

        if chat_id in message_queues:
            message_queues[chat_id].put(payload)

        # Opening a page counts as user-facing output, so finish_turn is
        # willing to end the turn.
        state["has_responded"] = True

        return (
            f"Opened course page '{resource.get('title', resource_id)}' "
            f"in the user's viewer. Section: {payload['section']}."
        )
