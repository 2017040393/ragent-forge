import pytest
from pydantic import BaseModel

from ragent_forge.app.models import AppConfig, GenerationConfig
from ragent_forge.app.services.context_service import (
    build_context_pack,
    build_generation_prompt,
)
from ragent_forge.app.services.generation_service import (
    GenerationService,
    NullGenerationProvider,
    OpenAIResponsesGenerationProvider,
)
from ragent_forge.app.services.search_service import SearchResult


def make_context_pack():
    return build_context_pack(
        "What is Agentic RAG?",
        [
            SearchResult(
                chunk_id="/knowledge/rag.md::chunk-0001",
                document_id="/knowledge/rag.md",
                source_path="/knowledge/rag.md",
                start_char=0,
                end_char=27,
                score=2.0,
                text="Agentic RAG uses planning.",
            )
        ],
    )


def test_null_generation_provider_returns_not_configured_without_answer() -> None:
    context_pack = make_context_pack()
    request = GenerationService().build_request(context_pack)

    result = NullGenerationProvider().generate(request)

    assert result.provider_name == "null"
    assert result.status == "not_configured"
    assert result.answer is None
    assert result.error is None
    assert result.metadata == {
        "reason": "No real generation provider is configured.",
    }


def test_generation_service_build_request_from_context_pack() -> None:
    context_pack = make_context_pack()

    request = GenerationService().build_request(context_pack)

    assert request.question == "What is Agentic RAG?"
    assert request.prompt == build_generation_prompt(context_pack)
    assert request.context_pack == context_pack
    assert "Generation is not implemented yet." not in request.prompt


def test_generation_service_uses_null_provider_by_default() -> None:
    context_pack = make_context_pack()

    result = GenerationService().generate(context_pack)

    assert result.provider_name == "null"
    assert result.status == "not_configured"
    assert result.answer is None


def test_null_generation_provider_does_not_include_fake_answer_text() -> None:
    context_pack = make_context_pack()
    request = GenerationService().build_request(context_pack)

    result = NullGenerationProvider().generate(request)

    assert result.answer is None
    assert "Agentic RAG uses planning" not in str(result.metadata)


def test_generation_service_from_config_uses_null_provider() -> None:
    service = GenerationService.from_config(
        AppConfig(generation=GenerationConfig(provider="null"))
    )

    assert isinstance(service.provider, NullGenerationProvider)


def test_generation_service_from_config_uses_openai_responses_provider(
) -> None:
    service = GenerationService.from_config(
        AppConfig(
            generation=GenerationConfig(
                provider="openai_responses",
                base_url="https://api.openai.com/v1/",
                model="gpt-4o-mini",
                api_key="super-secret-key",
                timeout_seconds=30,
                temperature=0.4,
                reasoning_effort="high",
            )
        )
    )

    assert isinstance(service.provider, OpenAIResponsesGenerationProvider)
    assert service.provider.base_url == "https://api.openai.com/v1"
    assert service.provider.model == "gpt-4o-mini"
    assert service.provider.timeout_seconds == 30
    assert service.provider.temperature == 0.4
    assert service.provider.reasoning_effort == "high"


def test_generation_service_from_config_missing_api_key_raises_clear_error() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid config file: generation.api_key is required "
            "when generation.provider is openai_responses"
        ),
    ):
        GenerationService.from_config(
            AppConfig(
                generation=GenerationConfig(
                    provider="openai_responses",
                    base_url="https://api.openai.com/v1",
                    model="gpt-4o-mini",
                    api_key=None,
                )
            )
        )


def test_generation_service_from_config_requires_base_url() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid config file: generation.base_url is required "
            "when generation.provider is openai_responses"
        ),
    ):
        GenerationService.from_config(
            AppConfig(
                generation=GenerationConfig(
                    provider="openai_responses",
                    base_url=None,
                    model="gpt-4o-mini",
                    api_key="super-secret-key",
                )
            )
        )


def test_generation_service_from_config_requires_model() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid config file: generation.model is required "
            "when generation.provider is openai_responses"
        ),
    ):
        GenerationService.from_config(
            AppConfig(
                generation=GenerationConfig(
                    provider="openai_responses",
                    base_url="https://api.openai.com/v1",
                    model=None,
                    api_key="super-secret-key",
                )
            )
        )


def test_generation_service_from_config_rejects_legacy_api_key_env_name() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid config file: generation.api_key_env is no longer supported; "
            "use generation.api_key instead"
        ),
    ):
        GenerationService.from_config(
            AppConfig(
                generation=GenerationConfig(
                    provider="openai_responses",
                    base_url="https://api.openai.com/v1",
                    model="gpt-4o-mini",
                    api_key=None,
                    api_key_env="OPENAI_API_KEY",
                )
            )
        )


def test_generation_service_from_config_rejects_unsupported_provider() -> None:
    class UnsupportedGenerationConfig(BaseModel):
        provider: str

    class UnsupportedAppConfig(BaseModel):
        generation: UnsupportedGenerationConfig

    config = UnsupportedAppConfig(
        generation=UnsupportedGenerationConfig(provider="openai")
    )

    with pytest.raises(ValueError, match="Unsupported generation provider: openai"):
        GenerationService.from_config(config)
