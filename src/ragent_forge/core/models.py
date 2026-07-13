from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeGuard

from pydantic import BaseModel, Field

SourceKind = Literal["document", "project_fact", "user_note", "session_memory"]
SourceAuthority = Literal["source", "user", "system", "derived"]
SourceLifecycle = Literal["regenerable", "user_owned", "session_scoped"]

SOURCE_KINDS: tuple[SourceKind, ...] = (
    "document",
    "project_fact",
    "user_note",
    "session_memory",
)
SOURCE_AUTHORITIES: tuple[SourceAuthority, ...] = (
    "source",
    "user",
    "system",
    "derived",
)
SOURCE_LIFECYCLES: tuple[SourceLifecycle, ...] = (
    "regenerable",
    "user_owned",
    "session_scoped",
)


def is_source_kind(value: object) -> TypeGuard[SourceKind]:
    return isinstance(value, str) and value in SOURCE_KINDS


def is_source_authority(value: object) -> TypeGuard[SourceAuthority]:
    return isinstance(value, str) and value in SOURCE_AUTHORITIES


def is_source_lifecycle(value: object) -> TypeGuard[SourceLifecycle]:
    return isinstance(value, str) and value in SOURCE_LIFECYCLES


def source_provenance_metadata(
    metadata: Mapping[str, object],
) -> dict[str, str]:
    result: dict[str, str] = {}
    source_kind = metadata.get("source_kind")
    if source_kind is not None:
        if not is_source_kind(source_kind):
            raise ValueError(f"Invalid source_kind: {source_kind!r}")
        result["source_kind"] = source_kind
    authority = metadata.get("authority")
    if authority is not None:
        if not is_source_authority(authority):
            raise ValueError(f"Invalid source authority: {authority!r}")
        result["authority"] = authority
    lifecycle = metadata.get("lifecycle")
    if lifecycle is not None:
        if not is_source_lifecycle(lifecycle):
            raise ValueError(f"Invalid source lifecycle: {lifecycle!r}")
        result["lifecycle"] = lifecycle
    for key in ("provenance", "freshness"):
        value = metadata.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Invalid source {key}: {value!r}")
        result[key] = value
    return result


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    text: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRef(BaseModel):
    document_id: str
    chunk_id: str
    source_path: str
    title: str | None = None
    source_kind: SourceKind = "document"
    provenance: str | None = None
    authority: SourceAuthority = "source"
    freshness: str | None = None
    lifecycle: SourceLifecycle = "regenerable"


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: SourceRef
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    name: str
    description: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagTrace(BaseModel):
    query: str
    steps: list[TraceStep] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    answer: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
