from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from ragent_forge.app.services.search_service import SearchResult

HybridSparseMode = Literal["bm25", "lexical"]
HybridDenseMode = Literal["semantic"]
HybridSparseMethod = Literal["bm25", "lexical_token_overlap"]
HybridDenseMethod = Literal["semantic_cosine_similarity"]


@dataclass(frozen=True)
class HybridSearchConfig:
    candidate_multiplier: int = 4
    min_candidate_limit: int = 20
    rrf_k: int = 60
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    sparse_mode: HybridSparseMode = "bm25"
    dense_mode: HybridDenseMode = "semantic"
    sparse_method: HybridSparseMethod = "bm25"
    dense_method: HybridDenseMethod = "semantic_cosine_similarity"

    @property
    def lexical_weight(self) -> float:
        return self.sparse_weight

    @property
    def semantic_weight(self) -> float:
        return self.dense_weight


class SearchServiceProtocol(Protocol):
    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        ...

    def count_chunks(self) -> int:
        ...


@dataclass
class _FusionCandidate:
    chunk_id: str
    sparse_result: SearchResult | None = None
    dense_result: SearchResult | None = None
    sparse_rank: int | None = None
    dense_rank: int | None = None

    @property
    def sparse_score(self) -> float | None:
        if self.sparse_result is None:
            return None
        return self.sparse_result.score

    @property
    def dense_score(self) -> float | None:
        if self.dense_result is None:
            return None
        return self.dense_result.score

    @property
    def best_rank(self) -> int:
        ranks = [
            rank
            for rank in (self.sparse_rank, self.dense_rank)
            if rank is not None
        ]
        return min(ranks)

    @property
    def representative_result(self) -> SearchResult:
        if self.sparse_result is None and self.dense_result is None:
            raise ValueError("fusion candidate has no search result")
        if self.sparse_result is None:
            assert self.dense_result is not None
            return self.dense_result
        if self.dense_result is None:
            return self.sparse_result
        if (
            self.dense_rank is not None
            and self.sparse_rank is not None
            and self.dense_rank < self.sparse_rank
        ):
            return self.dense_result
        return self.sparse_result


class HybridSearchService:
    def __init__(
        self,
        sparse_search_service: SearchServiceProtocol | None = None,
        dense_search_service: SearchServiceProtocol | None = None,
        config: HybridSearchConfig | None = None,
        *,
        lexical_search_service: SearchServiceProtocol | None = None,
        semantic_search_service: SearchServiceProtocol | None = None,
    ) -> None:
        resolved_sparse_service = sparse_search_service or lexical_search_service
        resolved_dense_service = dense_search_service or semantic_search_service
        if resolved_sparse_service is None:
            raise TypeError("sparse_search_service is required")
        if resolved_dense_service is None:
            raise TypeError("dense_search_service is required")
        self.sparse_search_service: SearchServiceProtocol = resolved_sparse_service
        self.dense_search_service: SearchServiceProtocol = resolved_dense_service
        self.lexical_search_service: SearchServiceProtocol = resolved_sparse_service
        self.semantic_search_service: SearchServiceProtocol = resolved_dense_service
        self.config = config or HybridSearchConfig()

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        candidate_limit = self.candidate_limit_for(limit)
        sparse_results = self.sparse_search_service.search(query, candidate_limit)
        dense_results = self.dense_search_service.search(query, candidate_limit)
        candidates = self._fuse_candidates(sparse_results, dense_results)

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
        return self.sparse_search_service.count_chunks()

    def candidate_limit_for(self, limit: int) -> int:
        return max(
            limit * self.config.candidate_multiplier,
            self.config.min_candidate_limit,
        )

    def _fuse_candidates(
        self,
        sparse_results: list[SearchResult],
        dense_results: list[SearchResult],
    ) -> dict[str, _FusionCandidate]:
        candidates: dict[str, _FusionCandidate] = {}
        for rank, result in enumerate(sparse_results, start=1):
            candidate = candidates.setdefault(
                result.chunk_id,
                _FusionCandidate(chunk_id=result.chunk_id),
            )
            candidate.sparse_result = result
            candidate.sparse_rank = rank

        for rank, result in enumerate(dense_results, start=1):
            candidate = candidates.setdefault(
                result.chunk_id,
                _FusionCandidate(chunk_id=result.chunk_id),
            )
            candidate.dense_result = result
            candidate.dense_rank = rank
        return candidates

    def _hybrid_score(self, candidate: _FusionCandidate) -> float:
        score = 0.0
        if candidate.sparse_rank is not None:
            score += self.config.sparse_weight * (
                1.0 / (self.config.rrf_k + candidate.sparse_rank)
            )
        if candidate.dense_rank is not None:
            score += self.config.dense_weight * (
                1.0 / (self.config.rrf_k + candidate.dense_rank)
            )
        return score

    def _to_search_result(
        self,
        candidate: _FusionCandidate,
        hybrid_score: float,
    ) -> SearchResult:
        result = candidate.representative_result
        matched_modes: list[str] = []
        if candidate.sparse_rank is not None:
            matched_modes.append(self.config.sparse_mode)
        if candidate.dense_rank is not None:
            matched_modes.append(self.config.dense_mode)

        return SearchResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            source_path=result.source_path,
            start_char=result.start_char,
            end_char=result.end_char,
            score=hybrid_score,
            text=result.text,
            source_kind=result.source_kind,
            provenance=result.provenance,
            authority=result.authority,
            freshness=result.freshness,
            lifecycle=result.lifecycle,
            metadata={
                **_safe_representative_metadata(result.metadata),
                "retrieval_method": "hybrid_rrf",
                "fusion_method": "reciprocal_rank_fusion",
                "rrf_k": self.config.rrf_k,
                "sparse_method": self.config.sparse_method,
                "dense_method": self.config.dense_method,
                "matched_modes": matched_modes,
                "sparse_rank": candidate.sparse_rank,
                "dense_rank": candidate.dense_rank,
                "sparse_score": candidate.sparse_score,
                "dense_score": candidate.dense_score,
                "hybrid_score": hybrid_score,
                "sparse_weight": self.config.sparse_weight,
                "dense_weight": self.config.dense_weight,
            },
        )


def _safe_representative_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    excluded_fragments = (
        "api_key",
        "secret",
        "token",
        "authorization",
        "embedding",
        "vector",
        "text",
    )
    return {
        key: value
        for key, value in metadata.items()
        if not any(fragment in key.lower() for fragment in excluded_fragments)
    }
