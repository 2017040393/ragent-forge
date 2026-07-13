"""Lazy compatibility facade for the moved embedding provider adapter."""

from importlib import import_module
from typing import Any

EmbeddingProvider: Any
EmbeddingService: Any
NoEmbeddingProvider: Any
OpenAIEmbeddingsProvider: Any

__all__ = [
    "EmbeddingProvider",
    "EmbeddingService",
    "NoEmbeddingProvider",
    "OpenAIEmbeddingsProvider",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(
        import_module("ragent_forge.infrastructure.providers.embedding"),
        name,
    )
    globals()[name] = value
    return value
