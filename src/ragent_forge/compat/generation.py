from __future__ import annotations

from ragent_forge.app.models import AppConfig
from ragent_forge.app.ports import HttpPostClient, HttpStreamClient
from ragent_forge.app.services.generation_runtime import (
    GenerationProviderPort,
    GenerationStreamEvent,
    GenerationStreamEventType,
    NullGenerationProvider,
)
from ragent_forge.app.services.generation_runtime import (
    GenerationService as _GenerationService,
)
from ragent_forge.infrastructure.providers.openai_generation import (
    OpenAIResponsesGenerationProvider,
)

GenerationProvider = GenerationProviderPort


class GenerationService(_GenerationService):
    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        http_client: HttpPostClient | HttpStreamClient | None = None,
    ) -> GenerationService:
        generation = config.generation
        if generation.provider == "null":
            return cls(NullGenerationProvider())
        if generation.provider != "openai_responses":
            raise ValueError(f"Unsupported generation provider: {generation.provider}")
        if not generation.base_url:
            raise ValueError(
                "Invalid config file: generation.base_url is required "
                "when generation.provider is openai_responses"
            )
        if not generation.model:
            raise ValueError(
                "Invalid config file: generation.model is required "
                "when generation.provider is openai_responses"
            )
        if not generation.api_key:
            raise ValueError(
                "Invalid config file: generation.api_key is required "
                "when generation.provider is openai_responses"
            )
        return cls(
            OpenAIResponsesGenerationProvider(
                base_url=generation.base_url,
                model=generation.model,
                api_key=generation.api_key,
                timeout_seconds=generation.timeout_seconds,
                temperature=generation.temperature,
                reasoning_effort=generation.reasoning_effort,
                http_client=http_client,
            )
        )


__all__ = [
    "GenerationProvider",
    "GenerationService",
    "GenerationStreamEvent",
    "GenerationStreamEventType",
    "NullGenerationProvider",
    "OpenAIResponsesGenerationProvider",
]
