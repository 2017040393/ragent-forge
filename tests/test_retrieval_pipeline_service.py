from ragent_forge.app.services.retrieval_pipeline_service import (
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


def _candidate(chunk_id: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_path="knowledge/rag.md",
        score=score,
        text=f"text for {chunk_id}",
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
