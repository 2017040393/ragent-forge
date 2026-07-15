import pytest

from ragent_forge.app.services.retrieval_pipeline_service import (
    RankedPrefixTokenBudgetContextSelector,
    RetrievalEngine,
    RetrievalPipelineService,
)
from ragent_forge.core.retrieval.contracts import RetrievalCandidate


class FakeCandidateSearchService:
    def __init__(self, candidates: list[RetrievalCandidate]) -> None:
        self.candidates = candidates
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 10) -> list[RetrievalCandidate]:
        self.queries.append((query, limit))
        return self.candidates

    def count_chunks(self) -> int:
        return len(self.candidates)


def _candidate(
    chunk_id: str,
    score: float,
    *,
    text: str | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_path="knowledge/rag.md",
        score=score,
        text=text if text is not None else f"text for {chunk_id}",
    )


def test_pipeline_records_stage_order_deduplicates_and_selects_context() -> None:
    search_service = FakeCandidateSearchService(
        [
            _candidate("chunk-1", 1.0),
            _candidate("chunk-1", 0.9),
            _candidate("chunk-2", 0.8),
        ]
    )
    pipeline = RetrievalPipelineService(
        search_service,
        retrieval_mode="bm25",
        retrieval_method="bm25",
        snapshot_id="snapshot-1",
    )

    run = pipeline.run("  agent memory  ", limit=2)

    assert search_service.queries == [("agent memory", 2)]
    assert [stage.name for stage in run.stages] == [
        "normalize_query",
        "candidate_retrieval",
        "deduplicate",
        "rerank",
        "context_selection",
        "trace",
    ]
    assert run.candidate_count == 3
    assert run.result_chunk_ids == ["chunk-1", "chunk-2"]
    assert run.snapshot_id == "snapshot-1"
    assert run.stages[2].outputs == {"deduplicated_count": 2}
    assert run.stages[3].status == "skipped"
    assert run.stages[3].outputs == {"reranker": None}
    assert run.stages[4].outputs["selection_policy"] == "top_k_v1"
    assert run.stages[4].outputs["characters_per_token"] == 4
    assert run.stages[4].outputs["selected_context_chars"] == 32
    assert run.stages[4].outputs["estimated_context_tokens"] == 8
    assert run.stages[-1].outputs == {"traceable": True}


def test_pipeline_skips_candidate_retrieval_for_empty_normalized_query() -> None:
    search_service = FakeCandidateSearchService([_candidate("chunk-1", 1.0)])
    pipeline = RetrievalPipelineService(
        search_service,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
    )

    run = pipeline.run("   ", limit=5)

    assert search_service.queries == []
    assert run.results == []
    assert run.stages[1].status == "skipped"
    assert run.stages[1].error == "query is empty"


def test_engine_uses_injected_reranker_and_records_it() -> None:
    class ReverseReranker:
        name = "reverse-test-reranker"

        def rerank(
            self,
            query: str,
            candidates: list[RetrievalCandidate],
        ) -> list[RetrievalCandidate]:
            assert query == "agent memory"
            return list(reversed(candidates))

    engine = RetrievalEngine(
        FakeCandidateSearchService(
            [_candidate("chunk-1", 1.0), _candidate("chunk-2", 0.5)]
        ),
        retrieval_mode="bm25",
        retrieval_method="bm25",
        reranker=ReverseReranker(),
    )

    run = engine.run("agent memory", limit=2)

    assert run.result_chunk_ids == ["chunk-2", "chunk-1"]
    assert run.stages[3].status == "completed"
    assert run.stages[3].outputs == {"reranker": "reverse-test-reranker"}


def test_ranked_prefix_token_budget_selects_only_whole_prefix_chunks() -> None:
    engine = RetrievalEngine(
        FakeCandidateSearchService(
            [
                _candidate("chunk-1", 1.0, text="a" * 12),
                _candidate("chunk-2", 0.9, text="b" * 16),
                _candidate("chunk-3", 0.8, text="c" * 13),
                _candidate("chunk-4", 0.7, text="d"),
            ]
        ),
        retrieval_mode="semantic",
        retrieval_method="semantic_cosine_similarity",
        context_selector=RankedPrefixTokenBudgetContextSelector(
            max_context_tokens=10,
            characters_per_token=4,
        ),
    )

    run = engine.run("agent memory", limit=4)

    assert run.result_chunk_ids == ["chunk-1", "chunk-2"]
    assert run.stages[4].outputs == {
        "selected_count": 2,
        "selected_chunk_ids": ["chunk-1", "chunk-2"],
        "selected_context_chars": 28,
        "estimated_context_tokens": 7,
        "selection_policy": "ranked_prefix_token_budget_v1",
        "max_context_tokens": 10,
        "characters_per_token": 4,
        "max_context_chars": 40,
    }


@pytest.mark.parametrize(
    ("max_context_tokens", "characters_per_token"),
    [(0, 4), (10, 0)],
)
def test_ranked_prefix_token_budget_rejects_invalid_configuration(
    max_context_tokens: int,
    characters_per_token: int,
) -> None:
    with pytest.raises(ValueError):
        RankedPrefixTokenBudgetContextSelector(
            max_context_tokens=max_context_tokens,
            characters_per_token=characters_per_token,
        )
