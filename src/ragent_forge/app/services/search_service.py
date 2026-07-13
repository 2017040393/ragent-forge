from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from ragent_forge.app.ports import ChunkReader
from ragent_forge.app.services.prepared_retrieval import (
    PreparedBM25Chunk,
    PreparedStateCache,
)
from ragent_forge.core.retrieval.contracts import (
    MetadataRecord,
    RetrievalCandidate,
)

SearchResult = RetrievalCandidate


class LexicalSearchService:
    def __init__(
        self,
        workspace: ChunkReader,
        *,
        prepared_state_cache: PreparedStateCache | None = None,
    ) -> None:
        self.workspace = workspace
        self.prepared_state_cache = prepared_state_cache or PreparedStateCache(
            tokenize
        )

    def count_chunks(self) -> int:
        return len(self.prepared_state_cache.prepare_chunks(self.workspace).records)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        prepared = self.prepared_state_cache.prepare_chunks(self.workspace)
        results: list[SearchResult] = []
        for chunk, text_tokens in prepared.lexical_chunks:
            text = str(chunk.get("text", ""))
            score = score_text(query_tokens, text_tokens)
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    chunk_id=str(chunk.get("chunk_id", "")),
                    document_id=str(chunk.get("document_id", "")),
                    source_path=str(chunk.get("source_path", "")),
                    start_char=_optional_int(chunk.get("start_char")),
                    end_char=_optional_int(chunk.get("end_char")),
                    score=score,
                    text=text,
                    metadata=_metadata(chunk.get("metadata")),
                    source_kind=chunk.get("source_kind", "document"),
                    provenance=chunk.get("provenance"),
                    authority=chunk.get("authority", "source"),
                    freshness=chunk.get("freshness"),
                    lifecycle=chunk.get("lifecycle", "regenerable"),
                )
            )

        return sorted(
            results,
            key=lambda result: (-result.score, result.chunk_id),
        )[:limit]


class BM25SearchService:
    def __init__(
        self,
        workspace: ChunkReader,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        prepared_state_cache: PreparedStateCache | None = None,
    ) -> None:
        self.workspace = workspace
        self.k1 = k1
        self.b = b
        self.prepared_state_cache = prepared_state_cache or PreparedStateCache(
            tokenize
        )

    def count_chunks(self) -> int:
        return len(self.prepared_state_cache.prepare_chunks(self.workspace).records)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        prepared = self.prepared_state_cache.prepare_chunks(self.workspace)
        chunks = prepared.bm25_chunks
        if not chunks:
            return []

        results: list[SearchResult] = []
        for chunk in chunks:
            score = self._score_chunk(
                query_tokens=query_tokens,
                chunk=chunk,
                document_frequencies=prepared.document_frequencies,
                document_count=len(chunks),
                average_document_length=prepared.average_document_length,
            )
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    chunk_id=str(chunk.record.get("chunk_id", "")),
                    document_id=str(chunk.record.get("document_id", "")),
                    source_path=str(chunk.record.get("source_path", "")),
                    start_char=_optional_int(chunk.record.get("start_char")),
                    end_char=_optional_int(chunk.record.get("end_char")),
                    score=score,
                    text=chunk.text,
                    metadata=_metadata(chunk.record.get("metadata")),
                    source_kind=chunk.record.get("source_kind", "document"),
                    provenance=chunk.record.get("provenance"),
                    authority=chunk.record.get("authority", "source"),
                    freshness=chunk.record.get("freshness"),
                    lifecycle=chunk.record.get("lifecycle", "regenerable"),
                )
            )

        return sorted(
            results,
            key=lambda result: (-result.score, result.chunk_id),
        )[:limit]

    def _score_chunk(
        self,
        *,
        query_tokens: list[str],
        chunk: PreparedBM25Chunk,
        document_frequencies: Counter[str],
        document_count: int,
        average_document_length: float,
    ) -> float:
        score = 0.0
        length_ratio = (
            chunk.length / average_document_length
            if average_document_length > 0
            else 0.0
        )
        for token in query_tokens:
            term_frequency = chunk.term_frequencies.get(token, 0)
            if term_frequency == 0:
                continue
            document_frequency = document_frequencies[token]
            idf = math.log(
                1
                + (
                    (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            )
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * length_ratio
            )
            if denominator == 0:
                continue
            score += idf * (
                term_frequency * (self.k1 + 1)
            ) / denominator
        return score


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def score_text(
    query_tokens: Sequence[str],
    text_tokens: Sequence[str],
) -> float:
    query_token_set = set(query_tokens)
    return float(sum(1 for token in text_tokens if token in query_token_set))


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _metadata(value: object) -> MetadataRecord:
    return MetadataRecord.from_value(value if isinstance(value, dict) else {})
