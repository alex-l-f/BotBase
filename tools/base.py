from abc import ABC, abstractmethod


class BaseTool(ABC):
    schema: dict

    @abstractmethod
    def execute(self, arguments: dict, context: dict):
        """Execute the tool and return a result (string or JSON-serialisable dict)."""
        ...
