import httpx
import pytest

from ragent_forge.app.models import GenerationRequest
from ragent_forge.app.services.context_service import build_context_pack
from ragent_forge.app.services.generation_service import (
    OpenAIResponsesGenerationProvider,
)
from ragent_forge.app.services.search_service import SearchResult


class FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, object]:
        return self.payload


class FakeHttpClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, headers: dict[str, str], json, timeout: int):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return self.response


class SequencedFakeHttpClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, headers: dict[str, str], json, timeout: int):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_generation_request() -> GenerationRequest:
    context_pack = build_context_pack(
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
    return GenerationRequest(
        question=context_pack.question,
        prompt=context_pack.generation_prompt,
        context_pack=context_pack,
    )


def test_openai_responses_generation_provider_posts_to_sanitized_endpoint() -> None:
    fake_client = FakeHttpClient(
        FakeResponse({"output_text": "Generated answer from responses"})
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1/",
        model="gpt-4o-mini",
        api_key="secret-key",
        http_client=fake_client,
    )

    result = provider.generate(make_generation_request())

    assert result.provider_name == "openai_responses"
    assert result.status == "success"
    assert result.answer == "Generated answer from responses"
    assert result.metadata == {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/responses",
    }
    assert fake_client.calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert fake_client.calls[0]["headers"] == {
        "Authorization": "Bearer secret-key",
        "Content-Type": "application/json",
    }
    assert fake_client.calls[0]["json"] == {
        "model": "gpt-4o-mini",
        "input": [
            {
                "role": "system",
                "content": (
                    "You are RAGentForge, a local retrieval-augmented assistant. "
                    "Use only the retrieved context."
                ),
            },
            {
                "role": "user",
                "content": make_generation_request().prompt,
            },
        ],
        "temperature": 0.2,
    }
    assert fake_client.calls[0]["timeout"] == 60


def test_openai_responses_generation_provider_includes_reasoning_effort_when_set(
    ) -> None:
    fake_client = FakeHttpClient(
        FakeResponse({"output_text": "Generated answer from responses"})
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1/",
        model="gpt-5.4",
        api_key="secret-key",
        reasoning_effort="low",
        http_client=fake_client,
    )

    provider.generate(make_generation_request())

    assert fake_client.calls[0]["json"]["reasoning"] == {"effort": "low"}


def test_openai_responses_generation_provider_parses_nested_output_text() -> None:
    fake_client = FakeHttpClient(
        FakeResponse(
            {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Nested generated answer",
                            }
                        ]
                    }
                ]
            }
        )
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://third-party.example.com/v1",
        model="some-responses-compatible-model",
        api_key="secret-key",
        http_client=fake_client,
    )

    result = provider.generate(make_generation_request())

    assert result.answer == "Nested generated answer"


def test_openai_responses_generation_provider_falls_back_to_untyped_output_text(
    ) -> None:
    fake_client = FakeHttpClient(
        FakeResponse(
            {
                "output": [
                    {
                        "content": [
                            {
                                "text": "Fallback generated answer",
                            }
                        ]
                    }
                ]
            }
        )
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://third-party.example.com/v1",
        model="some-responses-compatible-model",
        api_key="secret-key",
        http_client=fake_client,
    )

    result = provider.generate(make_generation_request())

    assert result.answer == "Fallback generated answer"


def test_openai_responses_generation_provider_rejects_unparseable_response() -> None:
    fake_client = FakeHttpClient(FakeResponse({"output": []}))
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://third-party.example.com/v1",
        model="some-responses-compatible-model",
        api_key="secret-key",
        http_client=fake_client,
    )

    with pytest.raises(
        RuntimeError,
        match="Generation provider failed: Could not parse response text",
    ):
        provider.generate(make_generation_request())


def test_openai_responses_generation_provider_wraps_http_errors_without_secret_key(
    ) -> None:
    fake_client = FakeHttpClient(
        FakeResponse({"error": "unauthorized"}, status_code=401)
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://third-party.example.com/v1",
        model="some-responses-compatible-model",
        api_key="secret-key",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="Generation provider failed: HTTP 401"):
        provider.generate(make_generation_request())


def test_openai_responses_generation_provider_retries_rate_limit_then_succeeds(
    ) -> None:
    fake_client = SequencedFakeHttpClient(
        [
            FakeResponse(
                {
                    "error": {
                        "message": "Too many pending requests, please retry later",
                        "type": "rate_limit_error",
                    }
                },
                status_code=429,
            ),
            FakeResponse({"output_text": "Generated answer after retry"}),
        ]
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://third-party.example.com/v1",
        model="some-responses-compatible-model",
        api_key="secret-key",
        http_client=fake_client,
    )

    result = provider.generate(make_generation_request())

    assert result.answer == "Generated answer after retry"
    assert len(fake_client.calls) == 2


def test_openai_responses_generation_provider_retries_ssl_eof_then_succeeds() -> None:
    fake_client = SequencedFakeHttpClient(
        [
            httpx.ConnectError(
                "[SSL: UNEXPECTED_EOF_WHILE_READING] "
                "EOF occurred in violation of protocol"
            ),
            FakeResponse({"output_text": "Generated answer after retry"}),
        ]
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://third-party.example.com/v1",
        model="some-responses-compatible-model",
        api_key="secret-key",
        http_client=fake_client,
    )

    result = provider.generate(make_generation_request())

    assert result.answer == "Generated answer after retry"
    assert len(fake_client.calls) == 2
