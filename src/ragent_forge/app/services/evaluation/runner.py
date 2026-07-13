from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from ragent_forge.app.services.evaluation.cases import load_cases
from ragent_forge.app.services.evaluation.contracts import (
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
    RetrievalEvalReport,
    RetrievalRunnerProtocol,
    SearchServiceProtocol,
    WorkspaceChunksProtocol,
)
from ragent_forge.app.services.evaluation.metrics import (
    compute_metrics as _compute_metrics,
)
from ragent_forge.app.services.evaluation.metrics import (
    ndcg_at as _ndcg_at,
)
from ragent_forge.app.services.evaluation.metrics import (
    precision_at as _precision_at,
)
from ragent_forge.app.services.evaluation.metrics import (
    round_metric as _round_metric,
)
from ragent_forge.app.services.evaluation.metrics import (
    summarize_stage_latencies as _summarize_stage_latencies,
)
from ragent_forge.app.services.evaluation.reporting import (
    classify_failure as _classify_failure,
)
from ragent_forge.app.services.evaluation.reporting import (
    compact_top_results as _compact_top_results,
)
from ragent_forge.app.services.evaluation.reporting import (
    dedupe_preserving_order as _dedupe_preserving_order,
)
from ragent_forge.app.services.evaluation.reporting import (
    source_path_matches as _source_path_matches,
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


class RetrievalEvalService:
    def load_cases(self, cases_path: str | Path) -> list[RetrievalEvalCase]:
        return load_cases(cases_path)

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
        representative_pipeline: list[dict[str, object]] = []
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
            if isinstance(search_service, RetrievalRunnerProtocol):
                retrieval_run = search_service.run(case.query, limit)
                search_results = retrieval_run.results
                if not representative_pipeline:
                    representative_pipeline = [
                        stage.model_dump(mode="json") for stage in retrieval_run.stages
                    ]
                result_metadata["retrieval_pipeline"] = [
                    stage.model_dump(mode="json") for stage in retrieval_run.stages
                ]
            else:
                search_results = search_service.search(case.query, limit)
            retrieval_latency_ms = (time.perf_counter() - retrieval_started_at) * 1000
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
            retrieval_pipeline=representative_pipeline,
            stage_latency_ms=_summarize_stage_latencies(results),
            results=results,
        )

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
            precision=_round_metric(_precision_at(relevant_result_ranks, limit)),
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
        return [
            {str(key): value for key, value in record.items()}
            for record in workspace.read_chunks()
        ]
    raise ValueError("Evidence-span evaluation requires an injected workspace adapter")


def _evidence_span_mapping_metadata(
    evidence_spans: list[EvidenceSpan],
    mapping_result: GoldChunkMappingResult,
) -> dict[str, Any]:
    return {
        "evidence_span_count": len(evidence_spans),
        "evidence_spans": [_evidence_span_summary(span) for span in evidence_spans],
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
                result.chunk_id in expected and result.chunk_id not in seen_chunk_ids
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
                _interval_union_length(overlaps) / (char_range[1] - char_range[0])
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
        normalized[index : index + size] for index in range(len(normalized) - size + 1)
    }
