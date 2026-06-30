from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ragent_forge.app.services.search_service import LexicalSearchService, SearchResult
from ragent_forge.app.workspace import LocalWorkspace


class AskRetrievalResult(BaseModel):
    question: str
    results: list[SearchResult]
    generation_status: Literal["not_implemented"] = "not_implemented"


class AskService:
    def __init__(self, workspace: LocalWorkspace) -> None:
        self.workspace = workspace
        self.search_service = LexicalSearchService(workspace)

    def retrieve_context(self, question: str, limit: int = 5) -> AskRetrievalResult:
        return AskRetrievalResult(
            question=question,
            results=self.search_service.search(question, limit),
        )

    def count_chunks(self) -> int:
        return self.search_service.count_chunks()
