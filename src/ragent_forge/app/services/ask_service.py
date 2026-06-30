from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ragent_forge.app.models import ContextPack, GenerationResult
from ragent_forge.app.services.context_service import build_context_pack
from ragent_forge.app.services.generation_service import GenerationService
from ragent_forge.app.services.search_service import LexicalSearchService, SearchResult
from ragent_forge.app.workspace import LocalWorkspace


class AskRetrievalResult(BaseModel):
    question: str
    results: list[SearchResult]
    context_pack: ContextPack
    generation_result: GenerationResult
    generation_status: Literal["not_implemented"] = "not_implemented"


class AskService:
    def __init__(
        self,
        workspace: LocalWorkspace,
        generation_service: GenerationService | None = None,
    ) -> None:
        self.workspace = workspace
        self.search_service = LexicalSearchService(workspace)
        self.generation_service = generation_service or GenerationService()

    def retrieve_context(
        self,
        question: str,
        limit: int = 5,
        max_context_chars: int = 4000,
    ) -> AskRetrievalResult:
        results = self.search_service.search(question, limit)
        context_pack = build_context_pack(question, results, max_context_chars)
        return AskRetrievalResult(
            question=question,
            results=results,
            context_pack=context_pack,
            generation_result=self.generation_service.generate(context_pack),
        )

    def count_chunks(self) -> int:
        return self.search_service.count_chunks()
