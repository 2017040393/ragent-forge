import json

import pytest

from ragent_forge.app.services.hybrid_search_service import (
    HybridSearchConfig,
    HybridSearchService,
)
from ragent_forge.app.services.search_service import SearchResult


class FakeSearchService:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        self.calls.append((query, limit))
        return self.results[:limit]

    def count_chunks(self) -> int:
        return 42


def make_result(
    chunk_id: str,
    *,
    score: float,
    text: str | None = None,
    metadata: dict[str, object] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_path=f"/knowledge/{chunk_id}.md",
        start_char=0,
        end_char=10,
        score=score,
        text=text or f"text for {chunk_id}",
        metadata=metadata or {},
    )


def test_hybrid_search_rrf_combines_modes_and_deduplicates_by_chunk_id() -> None:
    lexical = FakeSearchService(
        [
            make_result("chunk-a", score=3.0, text="lexical text"),
            make_result("chunk-b", score=1.0),
        ]
    )
    semantic = FakeSearchService(
        [
            make_result("chunk-c", score=0.9),
            make_result("chunk-a", score=0.7, text="semantic text"),
        ]
    )

    results = HybridSearchService(lexical, semantic).search("agent memory", limit=10)

    assert [result.chunk_id for result in results] == [
        "chunk-a",
        "chunk-c",
        "chunk-b",
    ]
    assert len(results) == 3
    assert results[0].text == "lexical text"
    assert results[0].metadata["matched_modes"] == ["lexical", "semantic"]
    assert results[1].metadata["matched_modes"] == ["semantic"]
    assert results[2].metadata["matched_modes"] == ["lexical"]


def test_hybrid_search_boosts_chunks_matched_by_both_modes() -> None:
    lexical = FakeSearchService(
        [
            make_result("chunk-a", score=3.0),
            make_result("chunk-b", score=2.0),
        ]
    )
    semantic = FakeSearchService(
        [
            make_result("chunk-b", score=0.9),
            make_result("chunk-c", score=0.8),
        ]
    )

    results = HybridSearchService(lexical, semantic).search("agent memory", limit=3)

    assert results[0].chunk_id == "chunk-b"
    assert results[0].score > results[1].score
    assert results[0].score == pytest.approx((1 / 62) + (1 / 61))


def test_hybrid_search_sorts_deterministically_after_score_ties() -> None:
    lexical = FakeSearchService([make_result("chunk-b", score=2.0)])
    semantic = FakeSearchService([make_result("chunk-a", score=0.9)])

    results = HybridSearchService(lexical, semantic).search("tie query", limit=2)

    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-b"]
    assert results[0].score == results[1].score


def test_hybrid_search_sets_fused_score_and_compact_metadata() -> None:
    lexical = FakeSearchService(
        [
            make_result(
                "chunk-a",
                score=3.0,
                metadata={
                    "media_type": "application/pdf",
                    "page_start": 7,
                    "page_end": 7,
                    "table_indices": [2],
                },
            )
        ]
    )
    semantic = FakeSearchService([make_result("chunk-a", score=0.7821)])

    result = HybridSearchService(lexical, semantic).search("agent memory", limit=1)[0]

    expected_score = (1 / 61) + (1 / 61)
    assert result.score == pytest.approx(expected_score)
    assert result.metadata == {
        "media_type": "application/pdf",
        "page_start": 7,
        "page_end": 7,
        "table_indices": [2],
        "retrieval_method": "hybrid_rrf",
        "fusion_method": "reciprocal_rank_fusion",
        "rrf_k": 60,
        "matched_modes": ["lexical", "semantic"],
        "lexical_rank": 1,
        "semantic_rank": 1,
        "lexical_score": 3.0,
        "semantic_score": 0.7821,
        "hybrid_score": pytest.approx(expected_score),
        "lexical_weight": 1.0,
        "semantic_weight": 1.0,
    }


def test_hybrid_search_respects_limit() -> None:
    lexical = FakeSearchService(
        [
            make_result("chunk-a", score=3.0),
            make_result("chunk-b", score=2.0),
        ]
    )
    semantic = FakeSearchService([make_result("chunk-c", score=0.9)])

    results = HybridSearchService(lexical, semantic).search("agent memory", limit=2)

    assert len(results) == 2


def test_hybrid_search_uses_candidate_limit_from_config() -> None:
    lexical = FakeSearchService([])
    semantic = FakeSearchService([])
    config = HybridSearchConfig(candidate_multiplier=3, min_candidate_limit=8)

    HybridSearchService(lexical, semantic, config).search("agent memory", limit=5)

    assert lexical.calls == [("agent memory", 15)]
    assert semantic.calls == [("agent memory", 15)]


def test_hybrid_search_uses_minimum_candidate_limit() -> None:
    lexical = FakeSearchService([])
    semantic = FakeSearchService([])
    config = HybridSearchConfig(candidate_multiplier=3, min_candidate_limit=20)

    HybridSearchService(lexical, semantic, config).search("agent memory", limit=5)

    assert lexical.calls == [("agent memory", 20)]
    assert semantic.calls == [("agent memory", 20)]


def test_hybrid_search_returns_empty_list_when_backends_find_no_results() -> None:
    results = HybridSearchService(
        FakeSearchService([]),
        FakeSearchService([]),
    ).search("missing", limit=10)

    assert results == []


def test_hybrid_search_rejects_negative_limit() -> None:
    service = HybridSearchService(FakeSearchService([]), FakeSearchService([]))

    with pytest.raises(ValueError, match="limit must be greater than or equal to 0"):
        service.search("agent memory", limit=-1)


def test_hybrid_search_delegates_count_chunks_to_lexical_service() -> None:
    service = HybridSearchService(FakeSearchService([]), FakeSearchService([]))

    assert service.count_chunks() == 42


def test_hybrid_search_metadata_omits_sensitive_and_large_values() -> None:
    lexical = FakeSearchService(
        [
            make_result(
                "chunk-a",
                score=3.0,
                text="full chunk text should not leak",
                metadata={
                    "api_key": "embedding-secret-key",
                    "embedding": [0.1, 0.2],
                    "text": "full chunk text should not leak",
                },
            )
        ]
    )
    semantic = FakeSearchService([])

    result = HybridSearchService(lexical, semantic).search("agent memory", limit=1)[0]

    metadata_json = json.dumps(result.metadata)
    assert "embedding-secret-key" not in metadata_json
    assert '"embedding"' not in metadata_json
    assert "full chunk text should not leak" not in metadata_json
