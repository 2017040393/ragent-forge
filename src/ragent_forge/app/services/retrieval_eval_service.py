from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ragent_forge.app.services.evidence_span_service import EvidenceSpan
from ragent_forge.app.services.gold_chunk_mapping_service import (
    GoldChunkMappingResult,
    GoldChunkMappingService,
)
from ragent_forge.app.services.hybrid_search_service import (
    HybridDenseMethod,
    HybridSparseMethod,
)
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.workspace import LocalWorkspace

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
    def search(self, query: str, limit: int) -> list[SearchResult]:
        ...


class WorkspaceChunksProtocol(Protocol):
    def read_chunks(self) -> list[dict[str, Any]]:
        ...


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


class RetrievalEvalService:
    def load_cases(self, cases_path: str | Path) -> list[RetrievalEvalCase]:
        path = Path(cases_path)
        if not path.is_file():
            raise FileNotFoundError(f"cases file not found: {path}")

        cases: list[RetrievalEvalCase] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in eval cases {path} at line "
                    f"{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid eval case at line {line_number}: expected object"
                )
            cases.append(self._case_from_payload(payload, line_number))

        if not cases:
            raise ValueError("no eval cases found")
        return cases

    def evaluate(
        self,
        *,
        cases: list[RetrievalEvalCase],
        search_service: SearchServiceProtocol,
        limit: int,
        retrieval_mode: Literal["lexical", "bm25", "semantic", "hybrid"],
        retrieval_method: str,
        cases_path: str | Path,
        workspace_path: str | Path,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        index_path: str | Path | None = None,
        fusion_method: str | None = None,
        rrf_k: int | None = None,
        sparse_method: HybridSparseMethod | None = None,
        dense_method: HybridDenseMethod | None = None,
        sparse_weight: float | None = None,
        dense_weight: float | None = None,
        lexical_weight: float | None = None,
        semantic_weight: float | None = None,
        workspace: WorkspaceChunksProtocol | None = None,
    ) -> RetrievalEvalReport:
        if limit < 1:
            raise ValueError("limit must be greater than 0")
        if not cases:
            raise ValueError("no eval cases found")

        chunk_records = _read_chunks_for_evidence_spans(
            cases=cases,
            workspace=workspace,
            workspace_path=workspace_path,
        )
        gold_chunk_mapping_service = GoldChunkMappingService()

        results: list[RetrievalEvalCaseResult] = []
        for case in cases:
            effective_expected_chunk_ids = list(case.expected_chunk_ids)
            result_metadata: dict[str, Any] = {}
            mapping_coverage: float | None = None
            if case.evidence_spans:
                mapping_result = gold_chunk_mapping_service.map(
                    case.evidence_spans,
                    chunk_records or [],
                )
                effective_expected_chunk_ids = _dedupe_preserving_order(
                    [
                        *case.expected_chunk_ids,
                        *mapping_result.expected_chunk_ids,
                    ]
                )
                result_metadata.update(
                    _evidence_span_mapping_metadata(
                        case.evidence_spans,
                        mapping_result,
                    )
                )
                mapping_coverage = _round_metric(
                    len(mapping_result.span_mappings) / len(case.evidence_spans)
                )

            retrieval_started_at = time.perf_counter()
            search_results = search_service.search(case.query, limit)
            retrieval_latency_ms = (
                time.perf_counter() - retrieval_started_at
            ) * 1000
            results.append(
                self._evaluate_case(
                    case=case,
                    expected_chunk_ids=effective_expected_chunk_ids,
                    search_results=search_results,
                    limit=limit,
                    retrieval_latency_ms=retrieval_latency_ms,
                    mapping_coverage=mapping_coverage,
                    metadata=result_metadata,
                )
            )

        passed_count = sum(1 for result in results if result.passed)
        failed_count = len(results) - passed_count
        return RetrievalEvalReport(
            retrieval_mode=retrieval_mode,
            retrieval_method=retrieval_method,
            limit=limit,
            case_count=len(results),
            passed_count=passed_count,
            failed_count=failed_count,
            metrics=_compute_metrics(results, limit),
            cases_path=str(Path(cases_path)),
            workspace=str(Path(workspace_path)),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            index_path=str(Path(index_path)) if index_path is not None else None,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
            sparse_method=sparse_method,
            dense_method=dense_method,
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            results=results,
        )

    def _case_from_payload(
        self,
        payload: dict[str, Any],
        line_number: int,
    ) -> RetrievalEvalCase:
        known_fields = {
            "id",
            "query",
            "expected_chunk_ids",
            "expected_source_paths",
            "evidence_spans",
            "notes",
        }
        case_payload = {
            key: value
            for key, value in payload.items()
            if key in known_fields
        }
        case_payload["metadata"] = {
            key: value
            for key, value in payload.items()
            if key not in known_fields
        }
        try:
            return RetrievalEvalCase.model_validate(case_payload)
        except ValidationError as exc:
            errors = "; ".join(
                _format_validation_error(error)
                for error in exc.errors()
            )
            raise ValueError(
                f"Invalid eval case at line {line_number}: {errors}"
            ) from exc

    def _evaluate_case(
        self,
        *,
        case: RetrievalEvalCase,
        expected_chunk_ids: list[str],
        search_results: list[SearchResult],
        limit: int,
        retrieval_latency_ms: float,
        mapping_coverage: float | None,
        metadata: dict[str, Any],
    ) -> RetrievalEvalCaseResult:
        expected_chunk_id_set = set(expected_chunk_ids)
        expected_source_paths = tuple(case.expected_source_paths)
        chunk_rank = _first_matching_rank(
            search_results,
            lambda result: result.chunk_id in expected_chunk_id_set,
        )
        source_rank = None
        if chunk_rank is None:
            source_rank = _first_matching_rank(
                search_results,
                lambda result: _source_path_matches(
                    result.source_path,
                    expected_source_paths,
                ),
            )

        rank = chunk_rank if chunk_rank is not None else source_rank
        matched_by: Literal["chunk_id", "source_path", "none"]
        if chunk_rank is not None:
            matched_by = "chunk_id"
        elif source_rank is not None:
            matched_by = "source_path"
        else:
            matched_by = "none"

        failure_type, failure_reason = _classify_failure(
            case=case,
            expected_chunk_ids=expected_chunk_ids,
            search_results=search_results,
            matched_by=matched_by,
            metadata=metadata,
        )
        retrieved_expected_chunk_count = len(
            expected_chunk_id_set.intersection(
                result.chunk_id for result in search_results
            )
        )
        expected_chunk_count = len(expected_chunk_ids)
        relevant_flags, expected_relevant_count = _result_relevance(
            search_results=search_results,
            expected_chunk_ids=expected_chunk_ids,
            expected_source_paths=case.expected_source_paths,
        )
        relevant_result_ranks = [
            rank
            for rank, is_relevant in enumerate(relevant_flags, start=1)
            if is_relevant
        ]
        relevant_retrieved_count = len(relevant_result_ranks)
        retrieved_context_chars = sum(len(result.text) for result in search_results)
        relevant_context_chars = sum(
            len(result.text)
            for result, is_relevant in zip(
                search_results,
                relevant_flags,
                strict=True,
            )
            if is_relevant
        )
        return RetrievalEvalCaseResult(
            id=case.id,
            query=case.query,
            passed=rank is not None,
            rank=rank,
            reciprocal_rank=(1 / rank) if rank is not None else 0.0,
            matched_by=matched_by,
            failure_type=failure_type,
            failure_reason=failure_reason,
            expected_chunk_ids=expected_chunk_ids,
            expected_source_paths=case.expected_source_paths,
            actual_chunk_ids=[result.chunk_id for result in search_results],
            actual_source_paths=[result.source_path for result in search_results],
            top_results=_compact_top_results(search_results),
            retrieved_count=len(search_results),
            expected_chunk_count=expected_chunk_count,
            relevant_retrieved_count=relevant_retrieved_count,
            relevant_result_ranks=relevant_result_ranks,
            recall=(
                _round_metric(retrieved_expected_chunk_count / expected_chunk_count)
                if expected_chunk_count > 0
                else 0.0
            ),
            precision=_round_metric(
                _precision_at(relevant_result_ranks, limit)
            ),
            ndcg=_round_metric(
                _ndcg_at(
                    relevant_result_ranks,
                    expected_relevant_count=expected_relevant_count,
                    k=limit,
                )
            ),
            evidence_coverage=_evidence_coverage(
                case.evidence_spans,
                search_results,
            ),
            mapping_coverage=mapping_coverage,
            context_evidence_density=(
                _round_metric(relevant_context_chars / retrieved_context_chars)
                if retrieved_context_chars > 0
                else 0.0
            ),
            duplicate_context_ratio=_round_metric(
                _duplicate_context_ratio(search_results)
            ),
            retrieval_latency_ms=_round_metric(retrieval_latency_ms),
            retrieved_context_chars=retrieved_context_chars,
            estimated_context_tokens=math.ceil(retrieved_context_chars / 4),
            metadata=metadata,
        )


def _read_chunks_for_evidence_spans(
    *,
    cases: list[RetrievalEvalCase],
    workspace: WorkspaceChunksProtocol | None,
    workspace_path: str | Path,
) -> list[dict[str, Any]] | None:
    if not any(case.evidence_spans for case in cases):
        return None
    if workspace is not None:
        return workspace.read_chunks()
    return LocalWorkspace(workspace_path).read_chunks()


def _evidence_span_mapping_metadata(
    evidence_spans: list[EvidenceSpan],
    mapping_result: GoldChunkMappingResult,
) -> dict[str, Any]:
    return {
        "evidence_span_count": len(evidence_spans),
        "evidence_spans": [
            _evidence_span_summary(span)
            for span in evidence_spans
        ],
        "mapped_expected_chunk_ids": mapping_result.expected_chunk_ids,
        "unmatched_span_ids": mapping_result.unmatched_span_ids,
        "span_mappings": [
            {
                "span_id": mapping.span_id,
                "matched_chunk_ids": mapping.matched_chunk_ids,
                "match_method": mapping.match_method,
                "metadata": mapping.metadata,
            }
            for mapping in mapping_result.span_mappings
        ],
    }


def _evidence_span_summary(span: EvidenceSpan) -> dict[str, Any]:
    return {
        "id": span.id,
        "source_path": span.source_path,
        "document_id": span.document_id,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "page_start": span.page_start,
        "page_end": span.page_end,
        "media_type": span.media_type,
    }


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _first_matching_rank(
    search_results: list[SearchResult],
    predicate: Callable[[SearchResult], bool],
) -> int | None:
    for rank, result in enumerate(search_results, start=1):
        if predicate(result):
            return rank
    return None


def _result_relevance(
    *,
    search_results: list[SearchResult],
    expected_chunk_ids: list[str],
    expected_source_paths: list[str],
) -> tuple[list[bool], int]:
    if expected_chunk_ids:
        expected = set(expected_chunk_ids)
        seen_chunk_ids: set[str] = set()
        relevance: list[bool] = []
        for result in search_results:
            is_relevant = (
                result.chunk_id in expected
                and result.chunk_id not in seen_chunk_ids
            )
            relevance.append(is_relevant)
            if is_relevant:
                seen_chunk_ids.add(result.chunk_id)
        return relevance, len(expected)

    matched_source_indexes: set[int] = set()
    relevance = []
    for result in search_results:
        matched_index = next(
            (
                index
                for index, source_path in enumerate(expected_source_paths)
                if index not in matched_source_indexes
                and _source_path_matches(result.source_path, (source_path,))
            ),
            None,
        )
        relevance.append(matched_index is not None)
        if matched_index is not None:
            matched_source_indexes.add(matched_index)
    return relevance, len(expected_source_paths)


def _precision_at(relevant_result_ranks: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be greater than 0")
    relevant_count = sum(1 for rank in relevant_result_ranks if rank <= k)
    return relevant_count / k


def _ndcg_at(
    relevant_result_ranks: list[int],
    *,
    expected_relevant_count: int,
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be greater than 0")
    ideal_relevant_count = min(expected_relevant_count, k)
    if ideal_relevant_count == 0:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank in relevant_result_ranks
        if rank <= k
    )
    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )
    return dcg / ideal_dcg


def _evidence_coverage(
    evidence_spans: list[EvidenceSpan],
    search_results: list[SearchResult],
) -> float | None:
    span_coverages: list[float] = []
    for span in evidence_spans:
        matching_results = [
            result
            for result in search_results
            if _source_path_matches(result.source_path, (span.source_path,))
        ]
        char_range = _evidence_span_char_range(span)
        if char_range is not None:
            overlaps = [
                overlap
                for result in matching_results
                for overlap in [_char_overlap_interval(char_range, result)]
                if overlap is not None
            ]
            span_coverages.append(
                _interval_union_length(overlaps)
                / (char_range[1] - char_range[0])
            )
            continue

        span_pages = _evidence_span_pages(span)
        if span_pages:
            covered_pages: set[int] = set()
            for result in matching_results:
                covered_pages.update(span_pages & _search_result_pages(result))
            span_coverages.append(len(covered_pages) / len(span_pages))

    if not span_coverages:
        return None
    return _round_metric(sum(span_coverages) / len(span_coverages))


def _evidence_span_char_range(span: EvidenceSpan) -> tuple[int, int] | None:
    if (
        span.start_char is None
        or span.end_char is None
        or span.start_char < 0
        or span.end_char <= span.start_char
    ):
        return None
    return span.start_char, span.end_char


def _char_overlap_interval(
    span_range: tuple[int, int],
    result: SearchResult,
) -> tuple[int, int] | None:
    result_range = _search_result_char_range(result)
    if result_range is None:
        return None
    start = max(span_range[0], result_range[0])
    end = min(span_range[1], result_range[1])
    if end <= start:
        return None
    return start, end


def _search_result_char_range(result: SearchResult) -> tuple[int, int] | None:
    start = result.start_char
    end = result.end_char
    if start is None:
        start = _metadata_int(result.metadata, "start_char")
    if end is None:
        end = _metadata_int(result.metadata, "end_char")
    if start is None or end is None or start < 0 or end <= start:
        return None
    return start, end


def _interval_union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged_length = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start > current_end:
            merged_length += current_end - current_start
            current_start, current_end = start, end
            continue
        current_end = max(current_end, end)
    return merged_length + current_end - current_start


def _evidence_span_pages(span: EvidenceSpan) -> set[int]:
    if span.page_start is None and span.page_end is None:
        return set()
    start = span.page_start if span.page_start is not None else span.page_end
    end = span.page_end if span.page_end is not None else span.page_start
    if start is None or end is None or start < 1 or end < start:
        return set()
    return set(range(start, end + 1))


def _search_result_pages(result: SearchResult) -> set[int]:
    page_numbers = result.metadata.get("page_numbers")
    if isinstance(page_numbers, list):
        pages = {
            page
            for page in page_numbers
            if isinstance(page, int) and not isinstance(page, bool) and page >= 1
        }
        if pages:
            return pages

    page_number = _metadata_int(result.metadata, "page_number")
    if page_number is not None and page_number >= 1:
        return {page_number}

    start = _metadata_int(result.metadata, "page_start")
    end = _metadata_int(result.metadata, "page_end")
    if start is None and end is None:
        return set()
    start = start if start is not None else end
    end = end if end is not None else start
    if start is None or end is None or start < 1 or end < start:
        return set()
    return set(range(start, end + 1))


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _duplicate_context_ratio(search_results: list[SearchResult]) -> float:
    shingle_sets = [
        shingles
        for result in search_results
        for shingles in [_text_shingles(result.text)]
        if shingles
    ]
    total_shingles = sum(len(shingles) for shingles in shingle_sets)
    if total_shingles == 0:
        return 0.0
    unique_shingles: set[str] = set()
    for shingles in shingle_sets:
        unique_shingles.update(shingles)
    return (total_shingles - len(unique_shingles)) / total_shingles


def _text_shingles(text: str, size: int = 20) -> set[str]:
    normalized = " ".join(text.casefold().split())
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {
        normalized[index : index + size]
        for index in range(len(normalized) - size + 1)
    }


def _source_path_matches(actual_path: str, expected_paths: tuple[str, ...]) -> bool:
    actual = _normalize_eval_path(actual_path)
    return any(
        actual == expected or actual.endswith(f"/{expected}")
        for expected in (_normalize_eval_path(path) for path in expected_paths)
    )


def _classify_failure(
    *,
    case: RetrievalEvalCase,
    expected_chunk_ids: list[str],
    search_results: list[SearchResult],
    matched_by: MatchedBy,
    metadata: dict[str, Any],
) -> tuple[FailureType | None, str | None]:
    if matched_by != "none":
        return None, None

    if not search_results:
        return "no_result", "No retrieval results returned."

    unmatched_span_ids = _metadata_string_list(metadata, "unmatched_span_ids")
    if case.evidence_spans and unmatched_span_ids and not expected_chunk_ids:
        return (
            "unmapped_evidence",
            "Evidence spans could not be mapped to current chunks.",
        )

    expected_source_paths = _expected_source_paths_for_failure(case)
    if expected_source_paths and not any(
        _source_path_matches(result.source_path, expected_source_paths)
        for result in search_results
    ):
        return (
            "missed_source",
            "Retrieved results did not include any expected source path.",
        )

    expected_chunk_id_set = set(expected_chunk_ids)
    no_expected_chunk_retrieved = bool(expected_chunk_id_set) and all(
        result.chunk_id not in expected_chunk_id_set for result in search_results
    )
    mapped_expected_chunk_ids = _metadata_string_list(
        metadata,
        "mapped_expected_chunk_ids",
    )
    evidence_source_paths = _evidence_span_source_paths(case)
    if (
        no_expected_chunk_retrieved
        and mapped_expected_chunk_ids
        and evidence_source_paths
        and any(
            _source_path_matches(result.source_path, evidence_source_paths)
            for result in search_results
        )
    ):
        return (
            "wrong_section",
            "Expected source was retrieved, but no expected chunk was found in top-k.",
        )

    if no_expected_chunk_retrieved:
        return (
            "low_rank",
            "Expected chunks were not found within the evaluated top-k results.",
        )

    return "unknown", "No deterministic failure heuristic matched."


def _expected_source_paths_for_failure(
    case: RetrievalEvalCase,
) -> tuple[str, ...]:
    if case.expected_source_paths:
        return tuple(case.expected_source_paths)
    return _evidence_span_source_paths(case)


def _evidence_span_source_paths(case: RetrievalEvalCase) -> tuple[str, ...]:
    return tuple(
        _dedupe_preserving_order([span.source_path for span in case.evidence_spans])
    )


def _metadata_string_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _normalize_eval_path(path: str) -> str:
    return path.strip().replace("\\", "/").rstrip("/")


def _compact_top_results(search_results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": result.chunk_id,
            "source_path": result.source_path,
            "score": result.score,
        }
        for rank, result in enumerate(search_results, start=1)
    ]


def _compute_metrics(
    results: list[RetrievalEvalCaseResult],
    limit: int,
) -> dict[str, float]:
    case_count = len(results)
    if case_count == 0:
        raise ValueError("no eval cases found")

    def hit_at(k: int) -> float:
        hits = sum(
            1
            for result in results
            if result.rank is not None and result.rank <= k
        )
        return _round_metric(hits / case_count)

    def precision_at(k: int) -> float:
        precision = sum(
            _precision_at(result.relevant_result_ranks, k)
            for result in results
        ) / case_count
        return _round_metric(precision)

    mrr = sum(result.reciprocal_rank for result in results) / case_count
    recall = sum(result.recall for result in results) / case_count
    ndcg = sum(result.ndcg for result in results) / case_count
    evidence_coverage, evidence_coverage_case_rate = _optional_metric_average(
        [result.evidence_coverage for result in results]
    )
    mapping_coverage, mapping_coverage_case_rate = _optional_metric_average(
        [result.mapping_coverage for result in results]
    )
    context_evidence_density = (
        sum(result.context_evidence_density for result in results) / case_count
    )
    duplicate_context_ratio = (
        sum(result.duplicate_context_ratio for result in results) / case_count
    )
    retrieval_latencies = [result.retrieval_latency_ms for result in results]
    avg_retrieval_latency_ms = (
        sum(retrieval_latencies) / case_count
    )
    avg_retrieved_count = sum(result.retrieved_count for result in results) / case_count
    avg_retrieved_context_chars = (
        sum(result.retrieved_context_chars for result in results) / case_count
    )
    avg_estimated_context_tokens = (
        sum(result.estimated_context_tokens for result in results) / case_count
    )
    return {
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@k": hit_at(limit),
        "precision@1": precision_at(1),
        "precision@3": precision_at(3),
        "precision@5": precision_at(5),
        "precision@k": precision_at(limit),
        "mrr": _round_metric(mrr),
        "recall@k": _round_metric(recall),
        "ndcg@k": _round_metric(ndcg),
        "evidence_coverage@k": _round_metric(evidence_coverage),
        "evidence_coverage_case_rate": _round_metric(
            evidence_coverage_case_rate
        ),
        "mapping_coverage": _round_metric(mapping_coverage),
        "mapping_coverage_case_rate": _round_metric(mapping_coverage_case_rate),
        "context_evidence_density": _round_metric(context_evidence_density),
        "duplicate_context_ratio": _round_metric(duplicate_context_ratio),
        "avg_retrieval_latency_ms": _round_metric(avg_retrieval_latency_ms),
        "retrieval_latency_p50_ms": _round_metric(
            _percentile(retrieval_latencies, 0.5)
        ),
        "retrieval_latency_p95_ms": _round_metric(
            _percentile(retrieval_latencies, 0.95)
        ),
        "avg_retrieved_count": _round_metric(avg_retrieved_count),
        "avg_retrieved_context_chars": _round_metric(avg_retrieved_context_chars),
        "avg_estimated_context_tokens": _round_metric(avg_estimated_context_tokens),
    }


def _optional_metric_average(values: list[float | None]) -> tuple[float, float]:
    available = [value for value in values if value is not None]
    if not available:
        return 0.0, 0.0
    return sum(available) / len(available), len(available) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if percentile < 0 or percentile > 1:
        raise ValueError("percentile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def _round_metric(value: float) -> float:
    return round(value, 4)


def _format_validation_error(error: Mapping[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "invalid value"))
    if location:
        return f"{location}: {message}"
    return message
