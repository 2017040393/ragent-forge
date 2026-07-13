from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
