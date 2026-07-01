import httpx
import pytest

from ragent_forge.app.models import GenerationRequest
from ragent_forge.app.services.context_service import build_context_pack
from ragent_forge.app.services.generation_service import (
    OpenAIResponsesGenerationProvider,
)
from ragent_forge.app.services.search_service import SearchResult


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> object:
        return self.payload


class FakeHttpClient:
    def __init__(self, payload: object | None = None, error: Exception | None = None):
        self.payload = payload if payload is not None else {"output_text": "answer"}
        self.error = error
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, headers: dict[str, str], json, timeout: int):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return FakeResponse(self.payload)


class SequencedFakeHttpClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, headers: dict[str, str], json, timeout: int):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_request() -> GenerationRequest:
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


def test_endpoint_without_trailing_slash_uses_responses_path() -> None:
    fake_client = FakeHttpClient({"output_text": "Top-level answer"})
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        timeout_seconds=45,
        http_client=fake_client,
    )

    result = provider.generate(make_request())

    assert fake_client.calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert fake_client.calls[0]["timeout"] == 45
    assert result.metadata == {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/responses",
    }
    assert "api_key" not in result.metadata
    assert "sk-test-secret" not in str(result.metadata)


def test_endpoint_with_trailing_slash_uses_responses_path() -> None:
    fake_client = FakeHttpClient({"output_text": "Top-level answer"})
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1/",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    provider.generate(make_request())

    assert provider.base_url == "https://api.openai.com/v1"
    assert fake_client.calls[0]["url"] == "https://api.openai.com/v1/responses"


def test_request_body_shape_includes_expected_fields() -> None:
    fake_client = FakeHttpClient({"output_text": "Top-level answer"})
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    provider.generate(make_request())

    body = fake_client.calls[0]["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.2
    assert isinstance(body["input"], list)
    assert body["input"][0]["role"] == "system"
    assert body["input"][1]["role"] == "user"
    assert "What is Agentic RAG?" in body["input"][1]["content"]
    assert "/knowledge/rag.md" in body["input"][1]["content"]
    assert "/knowledge/rag.md::chunk-0001" in body["input"][1]["content"]
    assert "Agentic RAG uses planning." in body["input"][1]["content"]
    assert "Generation is not implemented yet." not in body["input"][1]["content"]


def test_reasoning_field_included_when_configured() -> None:
    fake_client = FakeHttpClient({"output_text": "Top-level answer"})
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        reasoning_effort="low",
        http_client=fake_client,
    )

    provider.generate(make_request())

    assert fake_client.calls[0]["json"]["reasoning"] == {"effort": "low"}


def test_reasoning_field_omitted_when_not_configured() -> None:
    fake_client = FakeHttpClient({"output_text": "Top-level answer"})
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        reasoning_effort=None,
        http_client=fake_client,
    )

    provider.generate(make_request())

    assert "reasoning" not in fake_client.calls[0]["json"]


def test_parses_top_level_output_text() -> None:
    fake_client = FakeHttpClient({"output_text": "Top-level answer"})
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    result = provider.generate(make_request())

    assert result.answer == "Top-level answer"
    assert result.status == "success"
    assert result.provider_name == "openai_responses"


def test_parses_nested_output_text() -> None:
    fake_client = FakeHttpClient(
        {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "Nested answer"},
                    ]
                }
            ]
        }
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    result = provider.generate(make_request())

    assert result.answer == "Nested answer"


def test_parses_nested_text_without_type_fallback() -> None:
    fake_client = FakeHttpClient(
        {
            "output": [
                {
                    "content": [
                        {"text": "Fallback nested answer"},
                    ]
                }
            ]
        }
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    result = provider.generate(make_request())

    assert result.answer == "Fallback nested answer"


def test_combines_multiple_nested_text_items() -> None:
    fake_client = FakeHttpClient(
        {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "Part 1"},
                        {"type": "output_text", "text": " Part 2"},
                    ]
                }
            ]
        }
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    result = provider.generate(make_request())

    assert result.answer == "Part 1 Part 2"


@pytest.mark.parametrize("payload", [{}, {"output": []}, []])
def test_malformed_or_non_dict_response_fails_clearly(payload: object) -> None:
    fake_client = FakeHttpClient(payload)
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(
        RuntimeError,
        match="Generation provider failed: Could not parse response text",
    ):
        provider.generate(make_request())


def test_http_provider_error_is_sanitized() -> None:
    fake_client = FakeHttpClient(error=RuntimeError("bad key sk-test-secret"))
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="Generation provider failed") as exc_info:
        provider.generate(make_request())

    assert "sk-test-secret" not in str(exc_info.value)


def test_retry_failure_is_sanitized() -> None:
    fake_client = SequencedFakeHttpClient(
        [
            httpx.TransportError("network failed sk-test-secret"),
            httpx.TransportError("network failed sk-test-secret"),
            httpx.TransportError("network failed sk-test-secret"),
        ]
    )
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )
    provider.retry_delay_seconds = 0

    with pytest.raises(RuntimeError, match="Generation provider failed after") as exc:
        provider.generate(make_request())

    assert "sk-test-secret" not in str(exc.value)
    assert len(fake_client.calls) == provider.max_attempts


def test_result_metadata_does_not_contain_api_key() -> None:
    fake_client = FakeHttpClient({"output_text": "Top-level answer"})
    provider = OpenAIResponsesGenerationProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    result = provider.generate(make_request())

    assert "api_key" not in result.metadata
    assert "sk-test-secret" not in str(result.metadata)
