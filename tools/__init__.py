import importlib
import pkgutil
import logging

from .base import BaseTool
from .toolsets import TOOLSETS

log = logging.getLogger(__name__)

_registry: dict[str, BaseTool] = {}


def load_tools() -> None:
    """Scan the tools/ package and register every BaseTool subclass found."""
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name in ("base", "toolsets"):
            continue
        module = importlib.import_module(f".{module_name}", package=__name__)
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseTool)
                and attr is not BaseTool
            ):
                instance = attr()
                name = instance.schema["function"]["name"]
                _registry[name] = instance
                log.debug("Registered tool: %s", name)


def get_schemas(toolset=None) -> list:
    """Return tool schemas to pass to the LLM.

    toolset can be:
      None          – all registered tools
      str           – a named profile from toolsets.py
      list[str]     – an explicit list of tool names
    """
    if toolset is None:
        return [tool.schema for tool in _registry.values()]

    if isinstance(toolset, str):
        if toolset not in TOOLSETS:
            raise ValueError(
                f"Unknown toolset {toolset!r}. Available: {list(TOOLSETS)}"
            )
        names = TOOLSETS[toolset]
    else:
        names = list(toolset)

    schemas = []
    for name in names:
        if name in _registry:
            schemas.append(_registry[name].schema)
        else:
            log.warning("Toolset references unknown tool %r — skipping", name)
    return schemas


def dispatch(name: str, arguments: dict, context: dict):
    """Call the tool registered under *name* and return its result."""
    if name not in _registry:
        return f"ERROR: Unknown tool '{name}'"
    return _registry[name].execute(arguments, context)
