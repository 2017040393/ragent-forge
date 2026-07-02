from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragent_forge.app.services.search_service import SearchResult


@dataclass(frozen=True)
class HybridSearchConfig:
    candidate_multiplier: int = 4
    min_candidate_limit: int = 20
    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0


class SearchServiceProtocol(Protocol):
    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        ...

    def count_chunks(self) -> int:
        ...


@dataclass
class _FusionCandidate:
    chunk_id: str
    lexical_result: SearchResult | None = None
    semantic_result: SearchResult | None = None
    lexical_rank: int | None = None
    semantic_rank: int | None = None

    @property
    def lexical_score(self) -> float | None:
        if self.lexical_result is None:
            return None
        return self.lexical_result.score

    @property
    def semantic_score(self) -> float | None:
        if self.semantic_result is None:
            return None
        return self.semantic_result.score

    @property
    def best_rank(self) -> int:
        ranks = [
            rank
            for rank in (self.lexical_rank, self.semantic_rank)
            if rank is not None
        ]
        return min(ranks)

    @property
    def representative_result(self) -> SearchResult:
        if self.lexical_result is None and self.semantic_result is None:
            raise ValueError("fusion candidate has no search result")
        if self.lexical_result is None:
            return self.semantic_result  # type: ignore[return-value]
        if self.semantic_result is None:
            return self.lexical_result
        if (
            self.semantic_rank is not None
            and self.lexical_rank is not None
            and self.semantic_rank < self.lexical_rank
        ):
            return self.semantic_result
        return self.lexical_result


class HybridSearchService:
    def __init__(
        self,
        lexical_search_service: SearchServiceProtocol,
        semantic_search_service: SearchServiceProtocol,
        config: HybridSearchConfig | None = None,
    ) -> None:
        self.lexical_search_service = lexical_search_service
        self.semantic_search_service = semantic_search_service
        self.config = config or HybridSearchConfig()

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        candidate_limit = self.candidate_limit_for(limit)
        lexical_results = self.lexical_search_service.search(query, candidate_limit)
        semantic_results = self.semantic_search_service.search(query, candidate_limit)
        candidates = self._fuse_candidates(lexical_results, semantic_results)

        scored_candidates = [
            (
                self._hybrid_score(candidate),
                candidate.best_rank,
                candidate.chunk_id,
                candidate,
            )
            for candidate in candidates.values()
        ]
        scored_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        top_candidates = scored_candidates[:limit]
        return [
            self._to_search_result(candidate, hybrid_score)
            for hybrid_score, _best_rank, _chunk_id, candidate in top_candidates
        ]

    def count_chunks(self) -> int:
        return self.lexical_search_service.count_chunks()

    def candidate_limit_for(self, limit: int) -> int:
        return max(
            limit * self.config.candidate_multiplier,
            self.config.min_candidate_limit,
        )

    def _fuse_candidates(
        self,
        lexical_results: list[SearchResult],
        semantic_results: list[SearchResult],
    ) -> dict[str, _FusionCandidate]:
        candidates: dict[str, _FusionCandidate] = {}
        for rank, result in enumerate(lexical_results, start=1):
            candidate = candidates.setdefault(
                result.chunk_id,
                _FusionCandidate(chunk_id=result.chunk_id),
            )
            candidate.lexical_result = result
            candidate.lexical_rank = rank

        for rank, result in enumerate(semantic_results, start=1):
            candidate = candidates.setdefault(
                result.chunk_id,
                _FusionCandidate(chunk_id=result.chunk_id),
            )
            candidate.semantic_result = result
            candidate.semantic_rank = rank
        return candidates

    def _hybrid_score(self, candidate: _FusionCandidate) -> float:
        score = 0.0
        if candidate.lexical_rank is not None:
            score += self.config.lexical_weight * (
                1.0 / (self.config.rrf_k + candidate.lexical_rank)
            )
        if candidate.semantic_rank is not None:
            score += self.config.semantic_weight * (
                1.0 / (self.config.rrf_k + candidate.semantic_rank)
            )
        return score

    def _to_search_result(
        self,
        candidate: _FusionCandidate,
        hybrid_score: float,
    ) -> SearchResult:
        result = candidate.representative_result
        matched_modes: list[str] = []
        if candidate.lexical_rank is not None:
            matched_modes.append("lexical")
        if candidate.semantic_rank is not None:
            matched_modes.append("semantic")

        return SearchResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            source_path=result.source_path,
            start_char=result.start_char,
            end_char=result.end_char,
            score=hybrid_score,
            text=result.text,
            metadata={
                "retrieval_method": "hybrid_rrf",
                "fusion_method": "reciprocal_rank_fusion",
                "rrf_k": self.config.rrf_k,
                "matched_modes": matched_modes,
                "lexical_rank": candidate.lexical_rank,
                "semantic_rank": candidate.semantic_rank,
                "lexical_score": candidate.lexical_score,
                "semantic_score": candidate.semantic_score,
                "hybrid_score": hybrid_score,
                "lexical_weight": self.config.lexical_weight,
                "semantic_weight": self.config.semantic_weight,
            },
        )
