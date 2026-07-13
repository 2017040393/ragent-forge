"""Compatibility facade for the top-level composition root."""

from ragent_forge.composition import (
    RetrievalRuntime,
    build_embedding_service,
    build_generation_service,
    build_retrieval_runtime,
    build_session_service,
    build_text_generation_client,
)

__all__ = [
    "RetrievalRuntime",
    "build_embedding_service",
    "build_generation_service",
    "build_retrieval_runtime",
    "build_session_service",
    "build_text_generation_client",
]
