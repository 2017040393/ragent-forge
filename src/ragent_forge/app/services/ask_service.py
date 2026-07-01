from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ragent_forge.app.models import ContextPack, GenerationResult, SourceRef
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


class AskAnswerResult(BaseModel):
    question: str
    results: list[SearchResult]
    context_pack: ContextPack
    generation_result: GenerationResult
    answer: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


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
        generation_result = (
            self._skip_generation()
            if not results
            else self.generation_service.generate(context_pack)
        )
        return AskRetrievalResult(
            question=question,
            results=results,
            context_pack=context_pack,
            generation_result=generation_result,
        )

    def ask(
        self,
        question: str,
        limit: int = 5,
        max_context_chars: int = 4000,
    ) -> AskAnswerResult:
        retrieval = self.retrieve_context(question, limit, max_context_chars)
        return AskAnswerResult(
            question=retrieval.question,
            results=retrieval.results,
            context_pack=retrieval.context_pack,
            generation_result=retrieval.generation_result,
            answer=retrieval.generation_result.answer,
            sources=[
                SourceRef(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    source_path=result.source_path,
                )
                for result in retrieval.results
            ],
        )

    def count_chunks(self) -> int:
        return self.search_service.count_chunks()

    def _skip_generation(self) -> GenerationResult:
        return GenerationResult(
            provider_name=self.generation_service.provider.provider_name,
            status="skipped",
            answer=None,
            metadata={"skip_reason": "no_retrieved_context"},
        )
