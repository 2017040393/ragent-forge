from __future__ import annotations

import math
from typing import Any

from ragent_forge.app.ports import RetrievalWorkspace
from ragent_forge.app.services.embedding_service import EmbeddingService
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.services.vector_index_service import VectorIndexService


class SemanticSearchService:
    def __init__(
        self,
        workspace: RetrievalWorkspace,
        embedding_service: EmbeddingService | Any,
    ) -> None:
        self.workspace = workspace
        self.embedding_service = embedding_service
        self.vector_index_service = VectorIndexService(workspace)

    def count_chunks(self) -> int:
        return len(self.workspace.read_chunks())

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")
        if not query.strip():
            return []

        query_result = self.embedding_service.embed_texts([query])
        query_embedding = query_result.embeddings[0]
        index_records = self.vector_index_service.read_index()
        chunks = self.workspace.read_chunks()
        chunk_by_id = {
            str(chunk.get("chunk_id", "")): chunk
            for chunk in chunks
        }

        scored_results: list[SearchResult] = []
        for record in index_records:
            score = cosine_similarity(query_embedding, record.embedding)
            chunk = chunk_by_id.get(record.chunk_id)
            if chunk is None:
                continue
            scored_results.append(
                SearchResult(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    source_path=record.source_path,
                    start_char=record.start_char,
                    end_char=record.end_char,
                    score=score,
                    text=str(chunk.get("text", "")),
                    source_kind=record.source_kind,
                    provenance=record.provenance,
                    authority=record.authority,
                    freshness=record.freshness,
                    lifecycle=record.lifecycle,
                    metadata={
                        **record.metadata,
                        "retrieval_method": "semantic_cosine_similarity",
                        "embedding_provider": record.embedding_provider,
                        "embedding_model": record.embedding_model,
                    },
                )
            )

        return sorted(
            scored_results,
            key=lambda result: (-result.score, result.chunk_id),
        )[:limit]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match")
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot_product = sum(left * right for left, right in zip(a, b, strict=True))
    return dot_product / (norm_a * norm_b)
