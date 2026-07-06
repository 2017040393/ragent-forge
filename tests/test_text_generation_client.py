from typing import Any, TypedDict

import pytest

from ragent_forge.app.services.text_generation_client import (
    OpenAIResponsesTextGenerationClient,
)


class FakeHttpCall(TypedDict):
    url: str
    headers: dict[str, str]
    json: dict[str, Any]
    timeout: int


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeSseResponse:
    headers = {"content-type": "text/event-stream"}

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


class FakeHttpClient:
    def __init__(
        self,
        payload: object | None = None,
        error: Exception | None = None,
        response: object | None = None,
    ):
        self.payload = payload if payload is not None else {"output_text": "answer"}
        self.error = error
        self.response = response
        self.calls: list[FakeHttpCall] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> object:
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
        if self.response is not None:
            return self.response
        return FakeResponse(self.payload)


def test_openai_text_generation_client_posts_eval_prompt_and_parses_output_text(
) -> None:
    fake_client = FakeHttpClient({"output_text": "Generated JSON"})
    client = OpenAIResponsesTextGenerationClient(
        base_url="https://api.openai.com/v1/",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        timeout_seconds=45,
        temperature=0.4,
        reasoning_effort="low",
        http_client=fake_client,
    )

    text = client.generate_text("User prompt", system_prompt="System prompt")

    assert text == "Generated JSON"
    assert fake_client.calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert fake_client.calls[0]["headers"]["Authorization"] == (
        "Bearer sk-test-secret"
    )
    assert fake_client.calls[0]["timeout"] == 45
    body = fake_client.calls[0]["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.4
    assert body["reasoning"] == {"effort": "low"}
    assert body["input"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
    ]


def test_openai_text_generation_client_parses_nested_output_text() -> None:
    fake_client = FakeHttpClient(
        {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "Nested JSON"},
                    ]
                }
            ]
        }
    )
    client = OpenAIResponsesTextGenerationClient(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    text = client.generate_text("User prompt")

    assert text == "Nested JSON"
    assert fake_client.calls[0]["json"]["input"][0]["content"] == (
        "You are generating retrieval evaluation cases for a RAG system."
    )
    assert "reasoning" not in fake_client.calls[0]["json"]


def test_openai_text_generation_client_parses_sse_output_text_deltas() -> None:
    sse = "\n\n".join(
        [
            'event: response.output_text.delta\n'
            'data: {"type":"response.output_text.delta","delta":"{\\"items\\":["}',
            'event: response.output_text.delta\n'
            'data: {"type":"response.output_text.delta","delta":"]}"}',
            "event: response.completed\n"
            'data: {"type":"response.completed","status":"completed"}',
        ]
    )
    fake_client = FakeHttpClient(response=FakeSseResponse(sse))
    client = OpenAIResponsesTextGenerationClient(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    text = client.generate_text("User prompt")

    assert text == '{"items":[]}'


def test_openai_text_generation_client_prefers_sse_done_text() -> None:
    sse = "\n\n".join(
        [
            'event: response.output_text.delta\n'
            'data: {"type":"response.output_text.delta","delta":"partial"}',
            'event: response.output_text.done\n'
            'data: {"type":"response.output_text.done","text":"complete"}',
        ]
    )
    fake_client = FakeHttpClient(response=FakeSseResponse(sse))
    client = OpenAIResponsesTextGenerationClient(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    text = client.generate_text("User prompt")

    assert text == "complete"


def test_openai_text_generation_client_rejects_sse_without_text() -> None:
    fake_client = FakeHttpClient(
        response=FakeSseResponse(
            'event: response.completed\n'
            'data: {"type":"response.completed","status":"completed"}'
        )
    )
    client = OpenAIResponsesTextGenerationClient(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="Could not parse response text"):
        client.generate_text("User prompt")


def test_openai_text_generation_client_sanitizes_api_key_from_errors() -> None:
    fake_client = FakeHttpClient(error=RuntimeError("boom sk-test-secret"))
    client = OpenAIResponsesTextGenerationClient(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="Generation provider failed") as exc_info:
        client.generate_text("User prompt")

    assert "sk-test-secret" not in str(exc_info.value)
    assert "<hidden>" in str(exc_info.value)
