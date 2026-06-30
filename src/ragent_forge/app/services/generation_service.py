from __future__ import annotations

from typing import Protocol

from ragent_forge.app.models import (
    AppConfig,
    ContextPack,
    GenerationRequest,
    GenerationResult,
)


class GenerationProvider(Protocol):
    provider_name: str

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...


class NullGenerationProvider:
    provider_name = "null"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            provider_name=self.provider_name,
            status="not_configured",
            answer=None,
            metadata={
                "reason": "No real generation provider is configured.",
            },
        )


class GenerationService:
    def __init__(self, provider: GenerationProvider | None = None) -> None:
        self.provider = provider or NullGenerationProvider()

    @classmethod
    def from_config(cls, config: AppConfig) -> GenerationService:
        provider_name = config.generation.provider
        if provider_name == "null":
            return cls(NullGenerationProvider())
        raise ValueError(f"Unsupported generation provider: {provider_name}")

    def build_request(self, context_pack: ContextPack) -> GenerationRequest:
        return GenerationRequest(
            question=context_pack.question,
            prompt=context_pack.prompt_preview,
            context_pack=context_pack,
        )

    def generate(self, context_pack: ContextPack) -> GenerationResult:
        request = self.build_request(context_pack)
        return self.provider.generate(request)
