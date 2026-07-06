from __future__ import annotations

from typing import Any

import pytest

from ragent_forge.app.services.evidence_span_service import EvidenceSpan
from ragent_forge.app.services.gold_chunk_mapping_service import (
    GoldChunkMappingService,
)


def make_span(
    span_id: str = "span-001",
    *,
    source_path: str = "/workspace/docs/rag.md",
    start_char: int | None = 0,
    end_char: int | None = 100,
    page_start: int | None = None,
    page_end: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceSpan:
    return EvidenceSpan(
        id=span_id,
        source_path=source_path,
        document_id=source_path,
        start_char=start_char,
        end_char=end_char,
        text="evidence text",
        media_type="text/markdown",
        section_title=None,
        heading_path=(),
        block_types=("paragraph",),
        page_start=page_start,
        page_end=page_end,
        metadata=metadata or {},
    )


def make_chunk(
    chunk_id: str,
    *,
    source_path: str = "/workspace/docs/rag.md",
    start_char: int | None = 0,
    end_char: int | None = 100,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": source_path,
        "source_path": source_path,
        "start_char": start_char,
        "end_char": end_char,
        "metadata": metadata or {"source_path": source_path},
        "text": "chunk text",
    }


def test_maps_one_span_fully_contained_in_one_chunk() -> None:
    span = make_span(start_char=20, end_char=60)
    chunks = [make_chunk("chunk-1", start_char=0, end_char=100)]

    result = GoldChunkMappingService().map([span], chunks)

    assert result.expected_chunk_ids == ["chunk-1"]
    assert len(result.span_mappings) == 1
    mapping = result.span_mappings[0]
    assert mapping.span_id == "span-001"
    assert mapping.matched_chunk_ids == ["chunk-1"]
    assert mapping.match_method == "char_overlap"
    assert mapping.metadata["overlap_ratio"] == pytest.approx(1.0)
    assert result.unmatched_span_ids == []


def test_maps_one_span_crossing_two_chunks() -> None:
    span = make_span(start_char=50, end_char=150)
    chunks = [
        make_chunk("chunk-1", start_char=0, end_char=100),
        make_chunk("chunk-2", start_char=100, end_char=200),
    ]

    result = GoldChunkMappingService().map([span], chunks)

    assert result.expected_chunk_ids == ["chunk-1", "chunk-2"]
    assert result.span_mappings[0].matched_chunk_ids == ["chunk-1", "chunk-2"]
    assert result.span_mappings[0].metadata["overlap_ratios"] == {
        "chunk-1": pytest.approx(0.5),
        "chunk-2": pytest.approx(0.5),
    }


def test_does_not_map_when_source_path_differs() -> None:
    span = make_span(source_path="/workspace/docs/rag.md")
    chunks = [make_chunk("chunk-1", source_path="/workspace/docs/other.md")]

    result = GoldChunkMappingService().map([span], chunks)

    assert result.expected_chunk_ids == []
    assert result.span_mappings == []
    assert result.unmatched_span_ids == ["span-001"]


def test_respects_min_overlap_ratio() -> None:
    span = make_span(start_char=0, end_char=100)
    chunks = [make_chunk("chunk-1", start_char=80, end_char=100)]

    strict_result = GoldChunkMappingService(min_overlap_ratio=0.3).map(
        [span],
        chunks,
    )
    permissive_result = GoldChunkMappingService(min_overlap_ratio=0.2).map(
        [span],
        chunks,
    )

    assert strict_result.expected_chunk_ids == []
    assert strict_result.unmatched_span_ids == ["span-001"]
    assert permissive_result.expected_chunk_ids == ["chunk-1"]


def test_handles_windows_posix_path_normalization_and_suffix_matching() -> None:
    span = make_span(source_path=r"C:\repo\project\docs\rag.md")
    chunks = [make_chunk("chunk-1", source_path="docs/rag.md")]

    result = GoldChunkMappingService().map([span], chunks)

    assert result.expected_chunk_ids == ["chunk-1"]
    assert result.span_mappings[0].metadata["source_match"] == "suffix"


def test_maps_pdf_span_by_page_overlap_when_char_offsets_are_absent() -> None:
    span = make_span(
        source_path="/workspace/docs/paper.pdf",
        start_char=None,
        end_char=None,
        page_start=2,
        page_end=3,
    )
    chunks = [
        make_chunk(
            "chunk-1",
            source_path="/workspace/docs/paper.pdf",
            start_char=None,
            end_char=None,
            metadata={"source_path": "/workspace/docs/paper.pdf", "page_start": 3},
        )
    ]

    result = GoldChunkMappingService().map([span], chunks)

    assert result.expected_chunk_ids == ["chunk-1"]
    mapping = result.span_mappings[0]
    assert mapping.match_method == "page_overlap"
    assert mapping.metadata["page_overlap"] == [3]


def test_falls_back_to_page_overlap_when_char_overlap_cannot_match() -> None:
    span = make_span(
        source_path="/workspace/docs/paper.pdf",
        start_char=100,
        end_char=500,
        page_start=2,
        page_end=3,
    )
    chunks = [
        make_chunk(
            "chunk-1",
            source_path="/workspace/docs/paper.pdf",
            start_char=None,
            end_char=None,
            metadata={"source_path": "/workspace/docs/paper.pdf", "page_start": 3},
        )
    ]

    result = GoldChunkMappingService().map([span], chunks)

    assert result.expected_chunk_ids == ["chunk-1"]
    mapping = result.span_mappings[0]
    assert mapping.match_method == "page_overlap"
    assert mapping.metadata["page_overlap"] == [3]
    assert result.unmatched_span_ids == []


def test_does_not_fallback_to_source_only_matching_by_default() -> None:
    span = make_span(start_char=None, end_char=None)
    chunks = [
        make_chunk("chunk-1", start_char=None, end_char=None),
        make_chunk("chunk-2", start_char=None, end_char=None),
    ]

    result = GoldChunkMappingService().map([span], chunks)

    assert result.expected_chunk_ids == []
    assert result.span_mappings == []
    assert result.unmatched_span_ids == ["span-001"]


def test_source_only_fallback_matches_same_source_when_enabled() -> None:
    span = make_span(start_char=None, end_char=None)
    chunks = [
        make_chunk("chunk-1", start_char=None, end_char=None),
        make_chunk("chunk-2", start_char=None, end_char=None),
        make_chunk(
            "other",
            source_path="/workspace/docs/other.md",
            start_char=None,
            end_char=None,
        ),
    ]

    result = GoldChunkMappingService(allow_source_only_fallback=True).map(
        [span],
        chunks,
    )

    assert result.expected_chunk_ids == ["chunk-1", "chunk-2"]
    mapping = result.span_mappings[0]
    assert mapping.matched_chunk_ids == ["chunk-1", "chunk-2"]
    assert mapping.match_method == "source_only"


def test_deduplicates_chunk_ids_while_preserving_order() -> None:
    span = make_span(start_char=0, end_char=100)
    chunks = [
        make_chunk("chunk-1", start_char=0, end_char=60),
        make_chunk("chunk-2", start_char=40, end_char=100),
        make_chunk("chunk-1", start_char=0, end_char=60),
    ]

    result = GoldChunkMappingService().map([span], chunks)

    assert result.expected_chunk_ids == ["chunk-1", "chunk-2"]
    assert result.span_mappings[0].matched_chunk_ids == ["chunk-1", "chunk-2"]


def test_returns_unmatched_span_ids_for_unmapped_spans() -> None:
    matched = make_span("matched", start_char=0, end_char=100)
    unmatched = make_span(
        "unmatched",
        source_path="/workspace/docs/missing.md",
        start_char=0,
        end_char=100,
    )
    chunks = [make_chunk("chunk-1", start_char=0, end_char=100)]

    result = GoldChunkMappingService().map([matched, unmatched], chunks)

    assert result.expected_chunk_ids == ["chunk-1"]
    assert [mapping.span_id for mapping in result.span_mappings] == ["matched"]
    assert result.unmatched_span_ids == ["unmatched"]
