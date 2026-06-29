from __future__ import annotations

from typing import Any, Literal

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


class AskResult(BaseModel):
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    trace: RagTrace


class IngestResult(BaseModel):
    source_path: str
    documents: list[Document] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_files)


class WorkspaceStatus(BaseModel):
    root_path: str
    exists: bool
    has_chunks: bool
    has_summary: bool
    status: Literal["not_initialized", "incomplete", "ready"]
    chunks_path: str
    latest_summary_path: str
    summary: dict[str, Any] = Field(default_factory=dict)
    chunk_count_from_file: int | None = None
    missing_files: list[str] = Field(default_factory=list)
