from __future__ import annotations

import os
from typing import Any, Protocol

import httpx

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


class OpenAIResponsesGenerationProvider:
    provider_name = "openai_responses"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: int = 60,
        temperature: float = 0.2,
        http_client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.http_client = http_client or httpx

    def generate(self, request: GenerationRequest) -> GenerationResult:
        response = self._post(request)
        payload = self._read_payload(response)
        answer = self._parse_response_text(payload)
        return GenerationResult(
            provider_name=self.provider_name,
            status="success",
            answer=answer,
            metadata={
                "model": self.model,
                "base_url": self.base_url,
                "endpoint": "/responses",
            },
        )

    def _post(self, request: GenerationRequest) -> object:
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are RAGentForge, a local retrieval-augmented "
                        "assistant. Use only the retrieved context."
                    ),
                },
                {
                    "role": "user",
                    "content": request.prompt,
                },
            ],
            "temperature": self.temperature,
        }
        try:
            response = self.http_client.post(
                f"{self.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.timeout_seconds,
            )
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - wrapped consistently
            raise RuntimeError(f"Generation provider failed: {exc}") from exc

    def _read_payload(self, response: object) -> dict[str, object]:
        try:
            payload = response.json()  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - wrapped consistently
            raise RuntimeError(f"Generation provider failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                "Generation provider failed: Could not parse response text"
            )
        return payload

    def _parse_response_text(self, payload: dict[str, object]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        texts: list[str] = []
        output = payload.get("output")
        if isinstance(output, list):
            for output_item in output:
                if not isinstance(output_item, dict):
                    continue
                content = output_item.get("content")
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    content_type = content_item.get("type")
                    if content_type is None or content_type == "output_text":
                        texts.append(text)

        answer = "".join(texts).strip()
        if answer:
            return answer
        raise RuntimeError("Generation provider failed: Could not parse response text")


class GenerationService:
    def __init__(self, provider: GenerationProvider | None = None) -> None:
        self.provider = provider or NullGenerationProvider()

    @classmethod
    def from_config(cls, config: AppConfig) -> GenerationService:
        generation = config.generation
        provider_name = generation.provider
        if provider_name == "null":
            return cls(NullGenerationProvider())
        if provider_name == "openai_responses":
            base_url = generation.base_url
            model = generation.model
            if not base_url:
                raise ValueError(
                    "Invalid config file: generation.base_url is required "
                    "when generation.provider is openai_responses"
                )
            if not model:
                raise ValueError(
                    "Invalid config file: generation.model is required "
                    "when generation.provider is openai_responses"
                )
            api_key_env = generation.api_key_env
            if not api_key_env:
                raise ValueError(
                    "Invalid config file: generation.api_key_env is required "
                    "when generation.provider is openai_responses"
                )
            try:
                api_key = os.environ[api_key_env]
            except KeyError as exc:
                raise ValueError(
                    f"Missing API key environment variable: {api_key_env}"
                ) from exc
            if not api_key.strip():
                raise ValueError(
                    f"Missing API key environment variable: {api_key_env}"
                )
            return cls(
                OpenAIResponsesGenerationProvider(
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    timeout_seconds=generation.timeout_seconds,
                    temperature=generation.temperature,
                )
            )
        raise ValueError(f"Unsupported generation provider: {provider_name}")

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
        request = self.build_request(context_pack)
        return self.provider.generate(request)
