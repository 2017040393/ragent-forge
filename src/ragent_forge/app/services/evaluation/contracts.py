from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field, field_validator, model_validator

from ragent_forge.app.services.evidence_span_service import EvidenceSpan
from ragent_forge.app.services.hybrid_search_service import (
    HybridDenseMethod,
    HybridSparseMethod,
)
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.core.retrieval.contracts import RetrievalRun

MatchedBy = Literal["chunk_id", "source_path", "none"]
FailureType = Literal[
    "no_result",
    "unmapped_evidence",
    "missed_source",
    "wrong_section",
    "low_rank",
    "unknown",
]


class SearchServiceProtocol(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]: ...


class WorkspaceChunksProtocol(Protocol):
    def read_chunks(self) -> Sequence[Mapping[str, object]]: ...


@runtime_checkable
class RetrievalRunnerProtocol(Protocol):
    def run(self, query: str, limit: int) -> RetrievalRun: ...


class RetrievalEvalCase(BaseModel):
    id: str
    query: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_source_paths: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "query")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("expected_chunk_ids", "expected_source_paths")
    @classmethod
    def _dedupe_expected_values(cls, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if not normalized:
                raise ValueError("must contain non-empty strings")
            if normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped

    @model_validator(mode="after")
    def _requires_at_least_one_expected_value(self) -> Self:
        if (
            not self.expected_chunk_ids
            and not self.expected_source_paths
            and not self.evidence_spans
        ):
            raise ValueError(
                "expected_chunk_ids, expected_source_paths, or evidence_spans "
                "must be provided"
            )
        return self


class RetrievalEvalCaseResult(BaseModel):
    id: str
    query: str
    passed: bool
    rank: int | None = None
    reciprocal_rank: float = 0.0
    matched_by: MatchedBy
    failure_type: FailureType | None = None
    failure_reason: str | None = None
    expected_chunk_ids: list[str]
    expected_source_paths: list[str]
    actual_chunk_ids: list[str]
    actual_source_paths: list[str]
    top_results: list[dict[str, Any]]
    retrieved_count: int
    expected_chunk_count: int
    relevant_retrieved_count: int
    relevant_result_ranks: list[int]
    recall: float
    precision: float
    ndcg: float
    evidence_coverage: float | None = None
    mapping_coverage: float | None = None
    context_evidence_density: float
    duplicate_context_ratio: float
    retrieval_latency_ms: float
    retrieved_context_chars: int
    estimated_context_tokens: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalStageLatencySummary(BaseModel):
    sample_count: int
    average_ms: float
    p50_ms: float
    p95_ms: float


class RetrievalEvalReport(BaseModel):
    evaluation_type: Literal["retrieval"] = "retrieval"
    retrieval_mode: Literal["lexical", "bm25", "semantic", "hybrid"]
    retrieval_method: str
    limit: int
    case_count: int
    passed_count: int
    failed_count: int
    metrics: dict[str, float]
    cases_path: str
    workspace: str
    results: list[RetrievalEvalCaseResult]
    embedding_provider: str | None = None
    embedding_model: str | None = None
    index_path: str | None = None
    fusion_method: str | None = None
    rrf_k: int | None = None
    sparse_method: HybridSparseMethod | None = None
    dense_method: HybridDenseMethod | None = None
    sparse_weight: float | None = None
    dense_weight: float | None = None
    lexical_weight: float | None = None
    semantic_weight: float | None = None
    retrieval_pipeline: list[dict[str, object]] = Field(default_factory=list)
    stage_latency_ms: dict[str, RetrievalStageLatencySummary] = Field(
        default_factory=dict
    )
