from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from ragent_forge.core.models import SourceRef
from ragent_forge.core.retrieval.types import RetrievalMethod, RetrievalMode

MetadataRecord = dict[str, object]
"""JSON-compatible metadata carried with chunks and retrieval candidates."""

SourceRecord = SourceRef
"""Canonical source identity contract shared by retrieval and answer views."""


class ChunkRecord(TypedDict):
    """JSON-compatible chunk record exchanged by workspace adapters."""

    schema_version: int
    snapshot_id: str | None
    chunk_id: str
    document_id: str
    text: str
    source_path: str | None
    start_char: int | None
    end_char: int | None
    metadata: MetadataRecord


class RetrievalCandidate(BaseModel):
    """A ranked chunk candidate produced by a retrieval adapter."""

    chunk_id: str
    document_id: str
    source_path: str
    start_char: int | None = None
    end_char: int | None = None
    score: float
    text: str
    metadata: MetadataRecord = Field(default_factory=dict)


RetrievalStageName = Literal[
    "normalize_query",
    "candidate_retrieval",
    "deduplicate",
    "rerank",
    "context_selection",
    "trace",
]
RetrievalStageStatus = Literal["completed", "skipped", "failed"]


class RetrievalStageRecord(BaseModel):
    name: RetrievalStageName
    status: RetrievalStageStatus
    inputs: dict[str, object] = Field(default_factory=dict)
    outputs: dict[str, object] = Field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None


class RetrievalRun(BaseModel):
    query: str
    retrieval_mode: RetrievalMode
    retrieval_method: RetrievalMethod
    requested_limit: int
    candidate_count: int = 0
    result_count: int = 0
    result_chunk_ids: list[str] = Field(default_factory=list)
    results: list[RetrievalCandidate] = Field(default_factory=list)
    stages: list[RetrievalStageRecord] = Field(default_factory=list)
    snapshot_id: str | None = None
    metadata: MetadataRecord = Field(default_factory=dict)
