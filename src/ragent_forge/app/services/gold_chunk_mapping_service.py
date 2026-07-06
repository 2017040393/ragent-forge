from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from ragent_forge.app.services.evidence_span_service import EvidenceSpan

SourceMatchKind = Literal["exact", "suffix"]
MatchMethod = Literal["char_overlap", "page_overlap", "source_only"]
T = TypeVar("T")


@dataclass(frozen=True)
class SpanChunkMapping:
    span_id: str
    matched_chunk_ids: list[str]
    match_method: MatchMethod
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldChunkMappingResult:
    expected_chunk_ids: list[str]
    span_mappings: list[SpanChunkMapping]
    unmatched_span_ids: list[str]


@dataclass(frozen=True)
class _ChunkCandidate:
    chunk_id: str
    source_path: str
    source_match: SourceMatchKind
    record: Mapping[str, Any]
    metadata: Mapping[str, Any]


class GoldChunkMappingService:
    def __init__(
        self,
        *,
        min_overlap_ratio: float = 0.3,
        allow_source_only_fallback: bool = False,
    ) -> None:
        if min_overlap_ratio <= 0 or min_overlap_ratio > 1:
            raise ValueError("min_overlap_ratio must be greater than 0 and at most 1")
        self.min_overlap_ratio = min_overlap_ratio
        self.allow_source_only_fallback = allow_source_only_fallback

    def map(
        self,
        evidence_spans: list[EvidenceSpan],
        chunks: list[dict[str, Any]],
    ) -> GoldChunkMappingResult:
        span_mappings: list[SpanChunkMapping] = []
        unmatched_span_ids: list[str] = []
        expected_chunk_ids: list[str] = []

        for span in evidence_spans:
            mapping = self._map_span(span, chunks)
            if mapping is None:
                unmatched_span_ids.append(span.id)
                continue

            span_mappings.append(mapping)
            _extend_unique(expected_chunk_ids, mapping.matched_chunk_ids)

        return GoldChunkMappingResult(
            expected_chunk_ids=expected_chunk_ids,
            span_mappings=span_mappings,
            unmatched_span_ids=unmatched_span_ids,
        )

    def _map_span(
        self,
        span: EvidenceSpan,
        chunks: list[dict[str, Any]],
    ) -> SpanChunkMapping | None:
        candidates = _source_matched_candidates(span.source_path, chunks)
        if not candidates:
            return None

        char_range = _span_char_range(span)
        if char_range is not None:
            mapping = self._map_span_by_char_overlap(span, candidates, char_range)
            if mapping is not None:
                return mapping
            return None

        span_pages = _span_pages(span)
        if span_pages:
            mapping = self._map_span_by_page_overlap(span, candidates, span_pages)
            if mapping is not None:
                return mapping
            return None

        if self.allow_source_only_fallback:
            return _source_only_mapping(span, candidates)

        return None

    def _map_span_by_char_overlap(
        self,
        span: EvidenceSpan,
        candidates: list[_ChunkCandidate],
        char_range: tuple[int, int],
    ) -> SpanChunkMapping | None:
        span_start, span_end = char_range
        span_length = span_end - span_start
        matched_chunk_ids: list[str] = []
        overlap_ratios: dict[str, float] = {}
        source_matches: dict[str, SourceMatchKind] = {}

        for candidate in candidates:
            chunk_range = _chunk_char_range(candidate)
            if chunk_range is None:
                continue

            overlap = _range_overlap(
                span_start,
                span_end,
                chunk_range[0],
                chunk_range[1],
            )
            overlap_ratio = overlap / span_length
            if overlap_ratio < self.min_overlap_ratio:
                continue

            if candidate.chunk_id not in matched_chunk_ids:
                matched_chunk_ids.append(candidate.chunk_id)
                overlap_ratios[candidate.chunk_id] = overlap_ratio
                source_matches[candidate.chunk_id] = candidate.source_match

        if not matched_chunk_ids:
            return None

        metadata: dict[str, Any] = {
            "min_overlap_ratio": self.min_overlap_ratio,
            "overlap_ratios": overlap_ratios,
            "source_matches": source_matches,
        }
        if len(matched_chunk_ids) == 1:
            chunk_id = matched_chunk_ids[0]
            metadata["overlap_ratio"] = overlap_ratios[chunk_id]
            metadata["source_match"] = source_matches[chunk_id]

        return SpanChunkMapping(
            span_id=span.id,
            matched_chunk_ids=matched_chunk_ids,
            match_method="char_overlap",
            metadata=metadata,
        )

    def _map_span_by_page_overlap(
        self,
        span: EvidenceSpan,
        candidates: list[_ChunkCandidate],
        span_pages: set[int],
    ) -> SpanChunkMapping | None:
        matched_chunk_ids: list[str] = []
        page_overlaps: dict[str, list[int]] = {}
        source_matches: dict[str, SourceMatchKind] = {}

        for candidate in candidates:
            chunk_pages = _chunk_pages(candidate)
            if not chunk_pages:
                continue

            page_overlap = sorted(span_pages & chunk_pages)
            if not page_overlap:
                continue

            if candidate.chunk_id not in matched_chunk_ids:
                matched_chunk_ids.append(candidate.chunk_id)
                page_overlaps[candidate.chunk_id] = page_overlap
                source_matches[candidate.chunk_id] = candidate.source_match

        if not matched_chunk_ids:
            return None

        merged_page_overlap: list[int] = []
        for chunk_id in matched_chunk_ids:
            _extend_unique(merged_page_overlap, page_overlaps[chunk_id])

        metadata: dict[str, Any] = {
            "page_overlap": merged_page_overlap,
            "page_overlaps": page_overlaps,
            "source_matches": source_matches,
        }
        if len(matched_chunk_ids) == 1:
            metadata["source_match"] = source_matches[matched_chunk_ids[0]]

        return SpanChunkMapping(
            span_id=span.id,
            matched_chunk_ids=matched_chunk_ids,
            match_method="page_overlap",
            metadata=metadata,
        )


def _source_only_mapping(
    span: EvidenceSpan,
    candidates: list[_ChunkCandidate],
) -> SpanChunkMapping | None:
    matched_chunk_ids: list[str] = []
    source_matches: dict[str, SourceMatchKind] = {}
    for candidate in candidates:
        if candidate.chunk_id not in matched_chunk_ids:
            matched_chunk_ids.append(candidate.chunk_id)
            source_matches[candidate.chunk_id] = candidate.source_match

    if not matched_chunk_ids:
        return None

    metadata: dict[str, Any] = {
        "reason": "no usable char offsets or page metadata",
        "source_matches": source_matches,
    }
    if len(matched_chunk_ids) == 1:
        metadata["source_match"] = source_matches[matched_chunk_ids[0]]

    return SpanChunkMapping(
        span_id=span.id,
        matched_chunk_ids=matched_chunk_ids,
        match_method="source_only",
        metadata=metadata,
    )


def _source_matched_candidates(
    span_source_path: str,
    chunks: list[dict[str, Any]],
) -> list[_ChunkCandidate]:
    candidates: list[_ChunkCandidate] = []
    for chunk in chunks:
        metadata = _metadata_mapping(chunk.get("metadata"))
        chunk_id = _string_value(chunk.get("chunk_id"))
        if chunk_id is None:
            chunk_id = _string_value(chunk.get("id"))
        source_path = _string_value(chunk.get("source_path"))
        if source_path is None:
            source_path = _string_value(metadata.get("source_path"))
        if chunk_id is None or source_path is None:
            continue

        source_match = _source_match_kind(span_source_path, source_path)
        if source_match is None:
            continue

        candidates.append(
            _ChunkCandidate(
                chunk_id=chunk_id,
                source_path=source_path,
                source_match=source_match,
                record=chunk,
                metadata=metadata,
            )
        )
    return candidates


def _source_match_kind(
    span_source_path: str,
    chunk_source_path: str,
) -> SourceMatchKind | None:
    span_path = _normalize_source_path(span_source_path)
    chunk_path = _normalize_source_path(chunk_source_path)
    if span_path == chunk_path:
        return "exact"
    if _is_suffix_path_match(span_path, chunk_path):
        return "suffix"
    return None


def _normalize_source_path(source_path: str) -> str:
    return source_path.replace("\\", "/").rstrip("/")


def _is_suffix_path_match(left: str, right: str) -> bool:
    return left.endswith(f"/{right}") or right.endswith(f"/{left}")


def _span_char_range(span: EvidenceSpan) -> tuple[int, int] | None:
    if span.start_char is None or span.end_char is None:
        return None
    if span.end_char <= span.start_char:
        return None
    return span.start_char, span.end_char


def _chunk_char_range(candidate: _ChunkCandidate) -> tuple[int, int] | None:
    start_char = _int_value(candidate.record.get("start_char"))
    end_char = _int_value(candidate.record.get("end_char"))
    if start_char is None:
        start_char = _int_value(candidate.metadata.get("start_char"))
    if end_char is None:
        end_char = _int_value(candidate.metadata.get("end_char"))
    if start_char is None or end_char is None:
        return None
    if end_char <= start_char:
        return None
    return start_char, end_char


def _range_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _span_pages(span: EvidenceSpan) -> set[int]:
    pages = _page_set_from_range(span.page_start, span.page_end)
    if pages:
        return pages
    return _page_set_from_value(span.metadata.get("page_numbers"))


def _chunk_pages(candidate: _ChunkCandidate) -> set[int]:
    pages = _page_set_from_range(
        _int_value(candidate.metadata.get("page_start")),
        _int_value(candidate.metadata.get("page_end")),
    )
    if pages:
        return pages

    pages = _page_set_from_value(candidate.metadata.get("page_numbers"))
    if pages:
        return pages

    page_number = _int_value(candidate.metadata.get("page_number"))
    if page_number is None:
        return set()
    return {page_number}


def _page_set_from_range(
    page_start: int | None,
    page_end: int | None,
) -> set[int]:
    if page_start is None:
        return set()
    if page_end is None:
        page_end = page_start
    if page_end < page_start:
        return set()
    return set(range(page_start, page_end + 1))


def _page_set_from_value(value: Any) -> set[int]:
    if isinstance(value, int):
        return {value}
    if not isinstance(value, list | tuple | set):
        return set()

    pages: set[int] = set()
    for item in value:
        page_number = _int_value(item)
        if page_number is not None:
            pages.add(page_number)
    return pages


def _extend_unique(values: list[T], additions: list[T]) -> None:
    for addition in additions:
        if addition not in values:
            values.append(addition)


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _string_value(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
