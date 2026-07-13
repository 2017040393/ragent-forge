"""Lazy compatibility facade for the moved text-generation adapter."""

from importlib import import_module
from typing import Any

OpenAIResponsesTextGenerationClient: Any

__all__ = ["OpenAIResponsesTextGenerationClient"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(
        import_module(
            "ragent_forge.infrastructure.providers.openai_text_generation"
        ),
        name,
    )
    globals()[name] = value
    return value
