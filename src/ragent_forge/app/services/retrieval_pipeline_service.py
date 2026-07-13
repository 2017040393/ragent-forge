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


class RetrievalPipelineService:
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
    ) -> None:
        self.search_service = search_service
        self.retrieval_mode: RetrievalMode = retrieval_mode
        self.retrieval_method: RetrievalMethod = retrieval_method
        self.snapshot_id = snapshot_id

    def search(self, query: str, limit: int = 10) -> list[RetrievalCandidate]:
        return self.run(query, limit).results

    def count_chunks(self) -> int:
        return self.search_service.count_chunks()

    def run(self, query: str, limit: int = 10) -> RetrievalRun:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        stages: list[RetrievalStageRecord] = []
        normalized_query, normalize_stage = self._normalize_query(query)
        stages.append(normalize_stage)

        candidates: list[RetrievalCandidate]
        candidate_started = perf_counter()
        candidate_status: Literal["completed", "skipped"]
        if normalized_query:
            candidates = self.search_service.search(normalized_query, limit)
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
        deduplicated = _deduplicate(candidates)
        stages.append(
            RetrievalStageRecord(
                name="deduplicate",
                status="completed",
                inputs={"candidate_count": len(candidates)},
                outputs={"deduplicated_count": len(deduplicated)},
                latency_ms=_elapsed_ms(deduplicate_started),
            )
        )
        stages.append(
            RetrievalStageRecord(
                name="rerank",
                status="skipped",
                inputs={"candidate_count": len(deduplicated)},
                outputs={"reranker": None},
            )
        )

        context_started = perf_counter()
        results = deduplicated[:limit]
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

    def _normalize_query(
        self,
        query: str,
    ) -> tuple[str, RetrievalStageRecord]:
        started = perf_counter()
        normalized = query.strip()
        return normalized, RetrievalStageRecord(
            name="normalize_query",
            status="completed",
            inputs={"query": query},
            outputs={"normalized_query": normalized},
            latency_ms=_elapsed_ms(started),
        )


def _deduplicate(
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


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)
