import pytest
from pydantic import BaseModel

from ragent_forge.app.models import AppConfig, GenerationConfig
from ragent_forge.app.services.context_service import build_context_pack
from ragent_forge.app.services.generation_service import (
    GenerationService,
    NullGenerationProvider,
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
    assert request.prompt == context_pack.prompt_preview
    assert request.context_pack == context_pack


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
