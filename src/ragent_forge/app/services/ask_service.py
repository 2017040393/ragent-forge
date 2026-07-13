from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ragent_forge.app.models import ContextPack, GenerationResult, SourceRef
from ragent_forge.app.ports import ChunkReader
from ragent_forge.app.services.context_service import build_context_pack
from ragent_forge.app.services.generation_service import (
    GenerationService,
    GenerationStreamEvent,
)
from ragent_forge.app.services.search_service import LexicalSearchService, SearchResult
from ragent_forge.core.retrieval.contracts import RetrievalRun


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


@runtime_checkable
class StreamingGenerationServiceProtocol(Protocol):
    def stream_generate(
        self,
        context_pack: ContextPack,
    ) -> Iterator[GenerationStreamEvent]:
        ...


class SearchServiceProtocol(Protocol):
    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        ...

    def count_chunks(self) -> int:
        ...


class RetrievalEngineProtocol(Protocol):
    def run(self, query: str, limit: int = 10) -> RetrievalRun:
        ...

    def count_chunks(self) -> int:
        ...


class AskRetrievalResult(BaseModel):
    question: str
    results: list[SearchResult]
    context_pack: ContextPack
    generation_result: GenerationResult
    retrieval_run: RetrievalRun | None = None
    generation_status: Literal["not_implemented"] = "not_implemented"


class AskAnswerResult(BaseModel):
    question: str
    results: list[SearchResult]
    context_pack: ContextPack
    generation_result: GenerationResult
    answer: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    retrieval_run: RetrievalRun | None = None


AskStreamEventType = Literal["delta", "done"]


@dataclass(frozen=True)
class AskStreamEvent:
    type: AskStreamEventType
    text: str = ""
    result: AskAnswerResult | None = None


class AskService:
    def __init__(
        self,
        workspace: ChunkReader,
        generation_service: GenerationServiceProtocol | None = None,
        search_service: SearchServiceProtocol | None = None,
        retrieval_engine: RetrievalEngineProtocol | None = None,
        retrieval_pipeline: RetrievalEngineProtocol | None = None,
        retrieval_method: str = "lexical_token_overlap",
    ) -> None:
        self.workspace = workspace
        self.search_service: SearchServiceProtocol = (
            search_service or LexicalSearchService(workspace)
        )
        self.retrieval_method = retrieval_method
        if retrieval_engine is not None and retrieval_pipeline is not None:
            raise ValueError(
                "retrieval_engine and retrieval_pipeline cannot both be provided"
            )
        self.retrieval_engine = retrieval_engine or retrieval_pipeline
        self.retrieval_pipeline = self.retrieval_engine
        self.generation_service: GenerationServiceProtocol = (
            generation_service or GenerationService()
        )

    def retrieve_context(
        self,
        question: str,
        limit: int = 5,
        max_context_chars: int = 4000,
    ) -> AskRetrievalResult:
        results, retrieval_run = self._retrieve(question, limit)
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
            retrieval_run=retrieval_run,
        )

    def ask(
        self,
        question: str,
        limit: int = 5,
        max_context_chars: int = 4000,
    ) -> AskAnswerResult:
        retrieval = self.retrieve_context(question, limit, max_context_chars)
        return self._build_answer_result(
            question=retrieval.question,
            results=retrieval.results,
            context_pack=retrieval.context_pack,
            generation_result=retrieval.generation_result,
            retrieval_run=retrieval.retrieval_run,
        )

    def stream_answer(
        self,
        question: str,
        limit: int = 5,
        max_context_chars: int = 4000,
    ) -> Iterator[AskStreamEvent]:
        results, retrieval_run = self._retrieve(question, limit)
        context_pack = build_context_pack(
            question,
            results,
            max_context_chars,
            retrieval_method=self.retrieval_method,
        )
        if not results:
            yield AskStreamEvent(
                type="done",
                result=self._build_answer_result(
                    question=question,
                    results=results,
                    context_pack=context_pack,
                    generation_result=self._skip_generation(),
                    retrieval_run=retrieval_run,
                ),
            )
            return

        generation_result: GenerationResult | None = None
        for event in self._stream_generation(context_pack):
            if event.type == "delta":
                yield AskStreamEvent(type="delta", text=event.text)
            elif event.type == "done":
                generation_result = event.result

        if generation_result is None:
            generation_result = GenerationResult(
                provider_name=self.generation_service.provider.provider_name,
                status="failed",
                answer=None,
                error="Generation stream ended without a final result.",
            )
        yield AskStreamEvent(
            type="done",
            result=self._build_answer_result(
                question=question,
                results=results,
                context_pack=context_pack,
                generation_result=generation_result,
                retrieval_run=retrieval_run,
            ),
        )

    @classmethod
    def from_config(
        cls,
        workspace: ChunkReader,
        generation_service: GenerationServiceProtocol | None = None,
        search_service: SearchServiceProtocol | None = None,
        retrieval_engine: RetrievalEngineProtocol | None = None,
        retrieval_pipeline: RetrievalEngineProtocol | None = None,
        retrieval_method: str = "lexical_token_overlap",
    ) -> AskService:
        return cls(
            workspace=workspace,
            generation_service=generation_service or GenerationService(),
            search_service=search_service,
            retrieval_engine=retrieval_engine,
            retrieval_pipeline=retrieval_pipeline,
            retrieval_method=retrieval_method,
        )

    def count_chunks(self) -> int:
        if self.retrieval_engine is not None:
            return self.retrieval_engine.count_chunks()
        return self.search_service.count_chunks()

    def _retrieve(
        self,
        question: str,
        limit: int,
    ) -> tuple[list[SearchResult], RetrievalRun | None]:
        if self.retrieval_engine is None:
            return self.search_service.search(question, limit), None
        retrieval_run = self.retrieval_engine.run(question, limit)
        return retrieval_run.results, retrieval_run

    def _skip_generation(self) -> GenerationResult:
        return GenerationResult(
            provider_name=self.generation_service.provider.provider_name,
            status="skipped",
            answer=None,
            metadata={"skip_reason": "no_retrieved_context"},
        )

    def _stream_generation(
        self,
        context_pack: ContextPack,
    ) -> Iterator[GenerationStreamEvent]:
        if isinstance(self.generation_service, StreamingGenerationServiceProtocol):
            yield from self.generation_service.stream_generate(context_pack)
            return
        yield GenerationStreamEvent(
            type="done",
            result=self.generation_service.generate(context_pack),
        )

    def _build_answer_result(
        self,
        *,
        question: str,
        results: list[SearchResult],
        context_pack: ContextPack,
        generation_result: GenerationResult,
        retrieval_run: RetrievalRun | None = None,
    ) -> AskAnswerResult:
        return AskAnswerResult(
            question=question,
            results=results,
            context_pack=context_pack,
            generation_result=generation_result,
            answer=generation_result.answer,
            sources=[
                SourceRef(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    source_path=result.source_path,
                    source_kind=result.source_kind,
                    provenance=result.provenance,
                    authority=result.authority,
                    freshness=result.freshness,
                    lifecycle=result.lifecycle,
                )
                for result in results
            ],
            retrieval_run=retrieval_run,
        )
