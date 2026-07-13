from __future__ import annotations

from typing import Protocol, cast

from ragent_forge.app.models import AppConfig, EmbeddingResult
from ragent_forge.app.ports import HttpPostClient, HttpResponse


class EmbeddingProvider(Protocol):
    provider_name: str

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        ...


class NoEmbeddingProvider:
    provider_name = "none"

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        raise RuntimeError(
            "embedding provider is not configured. Set [embedding] "
            'provider = "openai_embeddings".'
        )


class OpenAIEmbeddingsProvider:
    provider_name = "openai_embeddings"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: int = 60,
        http_client: HttpPostClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        try:
            if self.http_client is None:
                raise RuntimeError("HTTP client is not configured")
            response = cast(
                HttpResponse,
                self.http_client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                },
                timeout=self.timeout_seconds,
                ),
            )
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - wrapped consistently
            raise self._wrap_error(str(exc)) from exc

        embeddings = self._parse_embeddings(payload, expected_count=len(texts))
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        return EmbeddingResult(
            provider_name=self.provider_name,
            model=self.model,
            embeddings=embeddings,
            usage=usage if isinstance(usage, dict) else {},
            metadata={
                "base_url": self.base_url,
                "endpoint": "/embeddings",
            },
        )

    def _parse_embeddings(
        self,
        payload: object,
        expected_count: int,
    ) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise self._wrap_error("Could not parse embedding response")
        data = payload.get("data")
        if not isinstance(data, list):
            raise self._wrap_error("Could not parse embedding response")
        if len(data) != expected_count:
            raise self._wrap_error(
                f"expected {expected_count} embeddings, received {len(data)}"
            )

        indexed_embeddings: list[tuple[int, list[float]]] = []
        for item in data:
            if not isinstance(item, dict):
                raise self._wrap_error("Could not parse embedding response")
            index = item.get("index")
            raw_embedding = item.get("embedding")
            if not isinstance(index, int) or not isinstance(raw_embedding, list):
                raise self._wrap_error("Could not parse embedding response")
            embedding = self._parse_embedding_values(raw_embedding)
            indexed_embeddings.append((index, embedding))

        indexed_embeddings.sort(key=lambda item: item[0])
        expected_indexes = list(range(expected_count))
        actual_indexes = [index for index, _ in indexed_embeddings]
        if actual_indexes != expected_indexes:
            raise self._wrap_error("Could not parse embedding response")
        return [embedding for _, embedding in indexed_embeddings]

    def _parse_embedding_values(self, values: list[object]) -> list[float]:
        embedding: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise self._wrap_error("embedding values must be numbers")
            embedding.append(float(value))
        return embedding

    def _sanitize_error_message(self, message: str) -> str:
        if self.api_key:
            return message.replace(self.api_key, "<hidden>")
        return message

    def _wrap_error(self, message: str) -> RuntimeError:
        return RuntimeError(
            f"Embedding provider failed: {self._sanitize_error_message(message)}"
        )


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self.provider = provider or NoEmbeddingProvider()

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        http_client: HttpPostClient | None = None,
    ) -> EmbeddingService:
        embedding = config.embedding
        if embedding.provider == "none":
            return cls(NoEmbeddingProvider())
        if embedding.provider == "openai_embeddings":
            if not embedding.base_url:
                raise ValueError(
                    "Invalid config file: embedding.base_url is required "
                    "when embedding.provider is openai_embeddings"
                )
            if not embedding.model:
                raise ValueError(
                    "Invalid config file: embedding.model is required "
                    "when embedding.provider is openai_embeddings"
                )
            if not embedding.api_key:
                raise ValueError(
                    "Invalid config file: embedding.api_key is required "
                    "when embedding.provider is openai_embeddings"
                )
            return cls(
                OpenAIEmbeddingsProvider(
                    base_url=embedding.base_url,
                    model=embedding.model,
                    api_key=embedding.api_key,
                    timeout_seconds=embedding.timeout_seconds,
                    http_client=http_client,
                )
            )
        raise ValueError(f"Unsupported embedding provider: {embedding.provider}")

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        return self.provider.embed_texts(texts)
