from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ragent_forge.core.models import (
    Document,
    DocumentChunk,
    RagTrace,
    RetrievedChunk,
    SourceRef,
    TraceStep,
)

__all__ = [
    "AppConfig",
    "AskResult",
    "ContextChunk",
    "ContextPack",
    "Document",
    "DocumentChunk",
    "EmbeddingConfig",
    "EmbeddingResult",
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResult",
    "IngestResult",
    "OperationTrace",
    "RagTrace",
    "RetrievedChunk",
    "SourceRef",
    "TraceListItem",
    "TraceListResult",
    "TraceStep",
    "WorkspaceStatus",
]


class ContextChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    start_char: int | None = None
    end_char: int | None = None
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPack(BaseModel):
    question: str
    retrieval_method: str = "lexical_token_overlap"
    context_chunks: list[ContextChunk] = Field(default_factory=list)
    total_context_chars: int
    generation_status: Literal["not_implemented"] = "not_implemented"
    prompt_preview: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def generation_prompt(self) -> str:
        return self.prompt_preview


class GenerationRequest(BaseModel):
    question: str
    prompt: str
    context_pack: ContextPack
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    provider_name: str
    status: Literal["not_configured", "success", "failed", "skipped"]
    answer: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    provider_name: str
    model: str
    embeddings: list[list[float]]
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationConfig(BaseModel):
    provider: Literal["null", "openai_responses"] = "null"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: int = 60
    temperature: float = 0.2
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = (
        None
    )


class EmbeddingConfig(BaseModel):
    provider: Literal["none", "openai_embeddings"] = "none"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: int = 60
    batch_size: int = 64


class AppConfig(BaseModel):
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class OperationTrace(BaseModel):
    trace_id: str
    operation: str
    status: Literal["success", "failed"]
    started_at: str
    finished_at: str | None = None
    steps: list[TraceStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceListItem(BaseModel):
    trace_id: str
    operation: str
    status: str
    started_at: str
    finished_at: str | None = None
    path: str


class TraceListResult(BaseModel):
    traces: list[TraceListItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    snapshot_id: str | None = None
