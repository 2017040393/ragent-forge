from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

from ragent_forge.app.models import ContextPack, GenerationResult, SourceRef
from ragent_forge.app.services.context_service import build_context_pack
from ragent_forge.app.services.generation_service import GenerationService
from ragent_forge.app.services.search_service import LexicalSearchService, SearchResult
from ragent_forge.app.workspace import LocalWorkspace


class ProviderNameProtocol(Protocol):
    @property
    def provider_name(self) -> str:
        ...


class GenerationServiceProtocol(Protocol):
    @property
    def provider(self) -> ProviderNameProtocol:
        ...

    def generate(self, context_pack: ContextPack) -> GenerationResult:
        ...


class SearchServiceProtocol(Protocol):
    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        ...

    def count_chunks(self) -> int:
        ...


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
        generation_service: GenerationServiceProtocol | None = None,
        search_service: SearchServiceProtocol | None = None,
        retrieval_method: str = "lexical_token_overlap",
    ) -> None:
        self.workspace = workspace
        self.search_service: SearchServiceProtocol = (
            search_service or LexicalSearchService(workspace)
        )
        self.retrieval_method = retrieval_method
        self.generation_service: GenerationServiceProtocol = (
            generation_service or GenerationService()
        )

    def retrieve_context(
        self,
        question: str,
        limit: int = 5,
        max_context_chars: int = 4000,
    ) -> AskRetrievalResult:
        results = self.search_service.search(question, limit)
        context_pack = build_context_pack(
            question,
            results,
            max_context_chars,
            retrieval_method=self.retrieval_method,
        )
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

    @classmethod
    def from_config(
        cls,
        workspace: LocalWorkspace,
        generation_service: GenerationServiceProtocol | None = None,
        search_service: SearchServiceProtocol | None = None,
        retrieval_method: str = "lexical_token_overlap",
    ) -> AskService:
        return cls(
            workspace=workspace,
            generation_service=generation_service or GenerationService(),
            search_service=search_service,
            retrieval_method=retrieval_method,
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
