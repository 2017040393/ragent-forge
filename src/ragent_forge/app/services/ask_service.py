from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ragent_forge.app.models import ContextPack
from ragent_forge.app.services.context_service import build_context_pack
from ragent_forge.app.services.search_service import LexicalSearchService, SearchResult
from ragent_forge.app.workspace import LocalWorkspace


class AskRetrievalResult(BaseModel):
    question: str
    results: list[SearchResult]
    context_pack: ContextPack
    generation_status: Literal["not_implemented"] = "not_implemented"


class AskService:
    def __init__(self, workspace: LocalWorkspace) -> None:
        self.workspace = workspace
        self.search_service = LexicalSearchService(workspace)

    def retrieve_context(
        self,
        question: str,
        limit: int = 5,
        max_context_chars: int = 4000,
    ) -> AskRetrievalResult:
        results = self.search_service.search(question, limit)
        return AskRetrievalResult(
            question=question,
            results=results,
            context_pack=build_context_pack(question, results, max_context_chars),
        )

    def count_chunks(self) -> int:
        return self.search_service.count_chunks()
