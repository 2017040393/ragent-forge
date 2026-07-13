from __future__ import annotations

from time import perf_counter
from typing import Literal, Protocol

from ragent_forge.core.retrieval.contracts import (
    RetrievalCandidate,
    RetrievalRun,
    RetrievalStageRecord,
)
from ragent_forge.core.retrieval.types import RetrievalMethod, RetrievalMode


class CandidateSearchService(Protocol):
    def search(self, query: str, limit: int = 10) -> list[RetrievalCandidate]:
        ...

    def count_chunks(self) -> int:
        ...


class QueryProcessor(Protocol):
    def normalize(self, query: str) -> str:
        ...


class CandidateDeduplicator(Protocol):
    def deduplicate(
        self,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        ...


class Reranker(Protocol):
    name: str

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        ...


class ContextSelector(Protocol):
    def select(
        self,
        candidates: list[RetrievalCandidate],
        limit: int,
    ) -> list[RetrievalCandidate]:
        ...


class DefaultQueryProcessor:
    def normalize(self, query: str) -> str:
        return query.strip()


class ChunkIdDeduplicator:
    def deduplicate(
        self,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        seen: set[str] = set()
        results: list[RetrievalCandidate] = []
        for candidate in candidates:
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            results.append(candidate)
        return results


class TopKContextSelector:
    def select(
        self,
        candidates: list[RetrievalCandidate],
        limit: int,
    ) -> list[RetrievalCandidate]:
        return candidates[:limit]


class RetrievalEngine:
    """Run retrieval as explicit, inspectable stages.

    Reranking is intentionally represented as a skipped stage until a reranker
    is configured. That keeps the trace contract stable while the v0.3 stages
    are introduced incrementally.
    """

    def __init__(
        self,
        search_service: CandidateSearchService,
        retrieval_mode: RetrievalMode,
        retrieval_method: RetrievalMethod,
        snapshot_id: str | None = None,
        query_processor: QueryProcessor | None = None,
        deduplicator: CandidateDeduplicator | None = None,
        reranker: Reranker | None = None,
        context_selector: ContextSelector | None = None,
    ) -> None:
        self.candidate_retriever = search_service
        self.retrieval_mode: RetrievalMode = retrieval_mode
        self.retrieval_method: RetrievalMethod = retrieval_method
        self.snapshot_id = snapshot_id
        self.query_processor = query_processor or DefaultQueryProcessor()
        self.deduplicator = deduplicator or ChunkIdDeduplicator()
        self.reranker = reranker
        self.context_selector = context_selector or TopKContextSelector()

    @property
    def search_service(self) -> CandidateSearchService:
        """Compatibility name for callers that still inspect the adapter."""
        return self.candidate_retriever

    def search(self, query: str, limit: int = 10) -> list[RetrievalCandidate]:
        return self.run(query, limit).results

    def count_chunks(self) -> int:
        return self.candidate_retriever.count_chunks()

    def run(self, query: str, limit: int = 10) -> RetrievalRun:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        stages: list[RetrievalStageRecord] = []
        normalize_started = perf_counter()
        normalized_query = self.query_processor.normalize(query)
        stages.append(
            RetrievalStageRecord(
                name="normalize_query",
                status="completed",
                inputs={"query": query},
                outputs={"normalized_query": normalized_query},
                latency_ms=_elapsed_ms(normalize_started),
            )
        )

        candidates: list[RetrievalCandidate]
        candidate_started = perf_counter()
        candidate_status: Literal["completed", "skipped"]
        if normalized_query:
            candidates = self.candidate_retriever.search(normalized_query, limit)
            candidate_status = "completed"
            candidate_error = None
        else:
            candidates = []
            candidate_status = "skipped"
            candidate_error = "query is empty"
        stages.append(
            RetrievalStageRecord(
                name="candidate_retrieval",
                status=candidate_status,
                inputs={"query": normalized_query, "limit": limit},
                outputs={"candidate_count": len(candidates)},
                latency_ms=_elapsed_ms(candidate_started),
                error=candidate_error,
            )
        )

        deduplicate_started = perf_counter()
        deduplicated = self.deduplicator.deduplicate(candidates)
        stages.append(
            RetrievalStageRecord(
                name="deduplicate",
                status="completed",
                inputs={"candidate_count": len(candidates)},
                outputs={"deduplicated_count": len(deduplicated)},
                latency_ms=_elapsed_ms(deduplicate_started),
            )
        )
        rerank_started = perf_counter()
        if self.reranker is None:
            ranked = deduplicated
            rerank_status: Literal["completed", "skipped"] = "skipped"
            rerank_outputs: dict[str, object] = {"reranker": None}
        else:
            ranked = self.reranker.rerank(normalized_query, deduplicated)
            rerank_status = "completed"
            rerank_outputs = {"reranker": self.reranker.name}
        stages.append(
            RetrievalStageRecord(
                name="rerank",
                status=rerank_status,
                inputs={"candidate_count": len(deduplicated)},
                outputs=rerank_outputs,
                latency_ms=_elapsed_ms(rerank_started),
            )
        )

        context_started = perf_counter()
        results = self.context_selector.select(ranked, limit)
        stages.append(
            RetrievalStageRecord(
                name="context_selection",
                status="completed",
                inputs={"requested_limit": limit},
                outputs={
                    "selected_count": len(results),
                    "selected_chunk_ids": [result.chunk_id for result in results],
                },
                latency_ms=_elapsed_ms(context_started),
            )
        )
        stages.append(
            RetrievalStageRecord(
                name="trace",
                status="completed",
                inputs={"stage_count": len(stages)},
                outputs={"traceable": True},
            )
        )
        return RetrievalRun(
            query=normalized_query,
            retrieval_mode=self.retrieval_mode,
            retrieval_method=self.retrieval_method,
            requested_limit=limit,
            candidate_count=len(candidates),
            result_count=len(results),
            result_chunk_ids=[result.chunk_id for result in results],
            results=results,
            stages=stages,
            snapshot_id=self.snapshot_id,
        )

RetrievalPipelineService = RetrievalEngine


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)
