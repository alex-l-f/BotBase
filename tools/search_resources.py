from .base import BaseTool


class SearchResources(BaseTool):
    schema = {
        "type": "function",
        "function": {
            "name": "search_resources",
            "description": (
                "Search the resource portal for useful mental health resources. "
                "If used multiple times, will avoid returning resources already acquired "
                "to allow for new resources to be discovered. Only call this if the user "
                "requests it, or if you ask them if searching is okay. This process can take a while."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The query, or queries, to search the resource portal with, "
                            "in addition to the collected user details. Should be a comma "
                            "separated list of queries."
                        ),
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "The language the retrieved resources should be in. Should match "
                            "the language of the conversation, unless otherwise specified, or "
                            "there is a lack of target language resources. Also supports 'all' "
                            "to search all resources, regardless of language."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }

    def execute(self, arguments: dict, context: dict):
        state = context["state"]
        state["done"] = False

        if "query" not in arguments:
            return "ERROR: Missing 'query' in search_resources"

        embedding_search = context["embedding_search"]
        existing_resources = context["existing_resources"]
        database = context["database"]
        fields_to_remove = context["fields_to_remove"]

        search_terms = arguments["query"]
        language = arguments.get("language", "all")
        existing_ids = {r["oid"] for r in existing_resources if "oid" in r}

        embedding_search.switch_provider(database)
        resource_scores = embedding_search.search(
            search_terms.split(","), language=language, k=10
        )

        resources = []
        for resource_id, score in resource_scores.items():
            if resource_id in existing_ids:
                continue
            details = embedding_search.get_resource_details(resource_id)
            if not details:
                continue
            for field in fields_to_remove:
                details.pop(field, None)
            details["portalURL"] = details["portal_url"]
            details["oid"] = details["id"]
            details["id"] = f"RES_{str(details['id']).zfill(5)}"
            details["explanation"] = f"Selected with relevance score: {score:.3f}"
            resources.append(details)

        search_summary = (
            f"Searched database {database} for: {search_terms.split(',')[0]}\n"
            f"Found {len(resources)} relevant resources"
        )
        existing_resources.extend(resources)
        state["search_summary"] = search_summary

        return {
            "resources": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "description": r["description"][:100] + "...",
                    "physical_address": r["physical_address"],
                }
                for r in resources
            ],
            "search_summary": search_summary,
        }
