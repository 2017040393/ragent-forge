import httpx
import pytest

from ragent_forge.app.services.embedding_service import OpenAIEmbeddingsProvider


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeHttpClient:
    def __init__(self, payload: object | None = None, error: Exception | None = None):
        self.payload = payload if payload is not None else make_embedding_payload()
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


class FlakyHttpClient(FakeHttpClient):
    def __init__(self, failures: list[Exception]) -> None:
        super().__init__()
        self.failures = list(failures)

    def post(self, url: str, *, headers: dict[str, str], json, timeout: int):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        if self.failures:
            raise self.failures.pop(0)
        return FakeResponse(self.payload)


def make_embedding_payload() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }


def test_openai_embeddings_provider_posts_to_endpoint_without_trailing_slash() -> None:
    fake_client = FakeHttpClient()
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        timeout_seconds=45,
        http_client=fake_client,
    )

    result = provider.embed_texts(["hello"])

    assert fake_client.calls[0]["url"] == "https://api.openai.com/v1/embeddings"
    assert fake_client.calls[0]["timeout"] == 45
    assert result.provider_name == "openai_embeddings"
    assert result.model == "text-embedding-3-small"
    assert result.embeddings == [[0.1, 0.2, 0.3]]
    assert result.usage == {"prompt_tokens": 3, "total_tokens": 3}
    assert "sk-test-secret" not in str(result.metadata)


def test_openai_embeddings_provider_posts_to_endpoint_with_trailing_slash() -> None:
    fake_client = FakeHttpClient()
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1/",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    provider.embed_texts(["hello"])

    assert provider.base_url == "https://api.openai.com/v1"
    assert fake_client.calls[0]["url"] == "https://api.openai.com/v1/embeddings"


def test_openai_embeddings_provider_request_body_shape() -> None:
    fake_client = FakeHttpClient(
        {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ],
            "usage": {},
        }
    )
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    provider.embed_texts(["text 1", "text 2"])

    assert fake_client.calls[0]["headers"] == {
        "Authorization": "Bearer sk-test-secret",
        "Content-Type": "application/json",
    }
    assert fake_client.calls[0]["json"] == {
        "model": "text-embedding-3-small",
        "input": ["text 1", "text 2"],
    }


def test_openai_embeddings_provider_preserves_response_order_by_index() -> None:
    fake_client = FakeHttpClient(
        {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
            "usage": {},
        }
    )
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    result = provider.embed_texts(["first", "second"])

    assert result.embeddings == [[0.1, 0.2], [0.4, 0.5]]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": []},
        {"data": [{"index": 0}]},
    ],
)
def test_openai_embeddings_provider_malformed_response_fails_clearly(
    payload: object,
) -> None:
    fake_client = FakeHttpClient(payload)
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="Embedding provider failed"):
        provider.embed_texts(["hello"])


def test_openai_embeddings_provider_count_mismatch_fails_clearly() -> None:
    fake_client = FakeHttpClient(
        {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
    )
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="expected 2 embeddings"):
        provider.embed_texts(["first", "second"])


def test_openai_embeddings_provider_non_number_embedding_values_fail_clearly() -> None:
    fake_client = FakeHttpClient(
        {"data": [{"index": 0, "embedding": [0.1, "bad"]}]}
    )
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="embedding values must be numbers"):
        provider.embed_texts(["hello"])


def test_openai_embeddings_provider_sanitizes_api_key_from_errors() -> None:
    fake_client = FakeHttpClient(error=RuntimeError("bad key sk-test-secret"))
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="Embedding provider failed") as exc_info:
        provider.embed_texts(["hello"])

    assert "sk-test-secret" not in str(exc_info.value)


def test_openai_embeddings_provider_retries_transport_errors() -> None:
    fake_client = FlakyHttpClient(
        [
            httpx.ReadTimeout("first timeout"),
            httpx.ConnectError("temporary connection failure"),
        ]
    )
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    result = provider.embed_texts(["hello"])

    assert len(fake_client.calls) == 3
    assert result.embeddings == [[0.1, 0.2, 0.3]]
    assert result.metadata["max_transport_attempts"] == 3


def test_openai_embeddings_provider_stops_after_transport_attempt_limit() -> None:
    fake_client = FlakyHttpClient(
        [
            httpx.ReadTimeout("timeout 1"),
            httpx.ReadTimeout("timeout 2"),
            httpx.ReadTimeout("timeout 3"),
        ]
    )
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="Embedding provider failed: timeout 3"):
        provider.embed_texts(["hello"])

    assert len(fake_client.calls) == 3


def test_openai_embeddings_provider_rejects_invalid_transport_attempts() -> None:
    with pytest.raises(ValueError, match="must be greater than 0"):
        OpenAIEmbeddingsProvider(
            base_url="https://api.openai.com/v1",
            model="text-embedding-3-small",
            api_key="sk-test-secret",
            http_client=FakeHttpClient(),
            max_transport_attempts=0,
        )


def test_openai_embeddings_provider_metadata_does_not_include_api_key() -> None:
    provider = OpenAIEmbeddingsProvider(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        api_key="sk-test-secret",
        http_client=FakeHttpClient(),
    )

    result = provider.embed_texts(["hello"])

    assert "api_key" not in result.metadata
    assert "sk-test-secret" not in str(result.metadata)
