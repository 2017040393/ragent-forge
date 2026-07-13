from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from ragent_forge.app.ports import ChunkReader
from ragent_forge.core.retrieval.contracts import (
    ChunkRecord,
    MetadataRecord,
    RetrievalCandidate,
)

SearchResult = RetrievalCandidate


class LexicalSearchService:
    def __init__(self, workspace: ChunkReader) -> None:
        self.workspace = workspace

    def count_chunks(self) -> int:
        return len(self.workspace.read_chunks())

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        results: list[SearchResult] = []
        for chunk in self.workspace.read_chunks():
            text = str(chunk.get("text", ""))
            score = score_text(query_tokens, tokenize(text))
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
    ) -> None:
        self.workspace = workspace
        self.k1 = k1
        self.b = b

    def count_chunks(self) -> int:
        return len(self.workspace.read_chunks())

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        chunks = [
            _BM25Chunk.from_record(chunk)
            for chunk in self.workspace.read_chunks()
        ]
        if not chunks:
            return []

        document_frequencies = _bm25_document_frequencies(chunks)
        average_document_length = (
            sum(chunk.length for chunk in chunks) / len(chunks)
        )
        results: list[SearchResult] = []
        for chunk in chunks:
            score = self._score_chunk(
                query_tokens=query_tokens,
                chunk=chunk,
                document_frequencies=document_frequencies,
                document_count=len(chunks),
                average_document_length=average_document_length,
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
        chunk: _BM25Chunk,
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


@dataclass(frozen=True)
class _BM25Chunk:
    record: ChunkRecord
    text: str
    tokens: list[str]
    term_frequencies: Counter[str]
    length: int

    @classmethod
    def from_record(cls, record: ChunkRecord) -> _BM25Chunk:
        text = str(record.get("text", ""))
        tokens = tokenize(text)
        return cls(
            record=record,
            text=text,
            tokens=tokens,
            term_frequencies=Counter(tokens),
            length=len(tokens),
        )


def _bm25_document_frequencies(chunks: list[_BM25Chunk]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for chunk in chunks:
        frequencies.update(set(chunk.tokens))
    return frequencies


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def score_text(query_tokens: list[str], text_tokens: list[str]) -> float:
    query_token_set = set(query_tokens)
    return float(sum(1 for token in text_tokens if token in query_token_set))


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _metadata(value: object) -> MetadataRecord:
    return MetadataRecord.from_value(value if isinstance(value, dict) else {})
