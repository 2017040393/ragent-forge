from __future__ import annotations

from typing import Any

from ragent_forge.app.services.evaluation.contracts import (
    FailureType,
    MatchedBy,
    RetrievalEvalCase,
)
from ragent_forge.app.services.search_service import SearchResult


def dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def source_path_matches(actual_path: str, expected_paths: tuple[str, ...]) -> bool:
    actual = _normalize_eval_path(actual_path)
    return any(
        actual == expected or actual.endswith(f"/{expected}")
        for expected in (_normalize_eval_path(path) for path in expected_paths)
    )


def classify_failure(
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
        source_path_matches(result.source_path, expected_source_paths)
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
            source_path_matches(result.source_path, evidence_source_paths)
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


def compact_top_results(
    search_results: list[SearchResult],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": result.chunk_id,
            "source_path": result.source_path,
            "score": result.score,
        }
        for rank, result in enumerate(search_results, start=1)
    ]


def _expected_source_paths_for_failure(
    case: RetrievalEvalCase,
) -> tuple[str, ...]:
    if case.expected_source_paths:
        return tuple(case.expected_source_paths)
    return _evidence_span_source_paths(case)


def _evidence_span_source_paths(case: RetrievalEvalCase) -> tuple[str, ...]:
    return tuple(
        dedupe_preserving_order([span.source_path for span in case.evidence_spans])
    )


def _metadata_string_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _normalize_eval_path(path: str) -> str:
    return path.strip().replace("\\", "/").rstrip("/")
