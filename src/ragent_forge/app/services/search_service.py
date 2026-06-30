from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ragent_forge.app.workspace import LocalWorkspace


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    start_char: int | None = None
    end_char: int | None = None
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LexicalSearchService:
    def __init__(self, workspace: LocalWorkspace) -> None:
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
                )
            )

        return sorted(
            results,
            key=lambda result: (-result.score, result.chunk_id),
        )[:limit]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def score_text(query_tokens: list[str], text_tokens: list[str]) -> float:
    query_token_set = set(query_tokens)
    return float(sum(1 for token in text_tokens if token in query_token_set))


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _metadata(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
