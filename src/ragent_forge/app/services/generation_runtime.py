from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

from ragent_forge.app.models import ContextPack, GenerationRequest, GenerationResult

GenerationStreamEventType = Literal["delta", "done"]


@dataclass(frozen=True)
class GenerationStreamEvent:
    type: GenerationStreamEventType
    text: str = ""
    result: GenerationResult | None = None


class GenerationProviderPort(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

    def stream_generate(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamEvent]:
        ...


class NullGenerationProvider:
    provider_name = "null"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            provider_name=self.provider_name,
            status="not_configured",
            answer=None,
            metadata={"reason": "No real generation provider is configured."},
        )

    def stream_generate(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamEvent]:
        yield GenerationStreamEvent(type="done", result=self.generate(request))


class GenerationService:
    def __init__(self, provider: GenerationProviderPort | None = None) -> None:
        self.provider = provider or NullGenerationProvider()

    def build_request(self, context_pack: ContextPack) -> GenerationRequest:
        return GenerationRequest(
            question=context_pack.question,
            prompt=context_pack.generation_prompt,
            context_pack=context_pack,
            metadata={
                "context_chunk_count": len(context_pack.context_chunks),
                "total_context_chars": context_pack.total_context_chars,
            },
        )

    def generate(self, context_pack: ContextPack) -> GenerationResult:
        return self.provider.generate(self.build_request(context_pack))

    def stream_generate(
        self,
        context_pack: ContextPack,
    ) -> Iterator[GenerationStreamEvent]:
        yield from self.provider.stream_generate(self.build_request(context_pack))
