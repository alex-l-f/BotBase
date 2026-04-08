from .base import BaseTool


class ExamineResource(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "examine_resource",
            "description": (
                "Returns all information about a selected resource. Use this tool to help "
                "provide more information about a resource if the user requests it, or if you "
                "think it might help you answer a user's query. This takes some time to complete, "
                "so use it only as necessary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {
                        "type": "string",
                        "description": "The exact ID of the resource you want more information on.",
                    }
                },
                "required": ["resource_id"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        context["state"]["done"] = False

        if "resource_id" not in arguments:
            return "ERROR: Missing 'resource_id' in examine_resource"

        resource_id = arguments["resource_id"]
        resource = next(
            (r for r in context["existing_resources"] if r["id"] == resource_id),
            None,
        )

        if resource is None:
            # Not yet retrieved. Look it up directly in the database
            oid = None
            if resource_id.startswith("RES_"):
                try:
                    oid = int(resource_id[4:])
                except ValueError:
                    pass
            if oid is not None:
                context["embedding_search"].switch_provider(context["database"])
                resource_details = context["embedding_search"].get_resource_details(oid)
                if resource_details:
                    for field in context["fields_to_remove"]:
                        resource_details.pop(field, None)
                    resource_details["portalURL"] = resource_details["portal_url"]
                    resource_details["oid"] = resource_details["id"]
                    resource_details["id"] = f"RES_{str(resource_details['id']).zfill(5)}"
                    resource_details["explanation"] = "Retrieved directly from database."
                    context["existing_resources"].append(resource_details)
                    resource = resource_details

        if resource is None:
            return f"ERROR: Resource with ID {resource_id} not found"

        return "\n".join(f"{key}: {value}" for key, value in resource.items())
