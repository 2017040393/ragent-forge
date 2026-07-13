"""Lazy compatibility facade for moved generation adapters."""

from importlib import import_module
from typing import Any

GenerationProvider: Any
GenerationService: Any
GenerationStreamEvent: Any
GenerationStreamEventType: Any
NullGenerationProvider: Any
OpenAIResponsesGenerationProvider: Any

__all__ = [
    "GenerationProvider",
    "GenerationService",
    "GenerationStreamEvent",
    "GenerationStreamEventType",
    "NullGenerationProvider",
    "OpenAIResponsesGenerationProvider",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module("ragent_forge.compat.generation"), name)
    globals()[name] = value
    return value
