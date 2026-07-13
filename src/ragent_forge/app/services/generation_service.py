from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, cast

from ragent_forge.app.models import (
    AppConfig,
    ContextPack,
    GenerationRequest,
    GenerationResult,
)
from ragent_forge.app.ports import (
    HttpPostClient,
    HttpStreamClient,
    HttpTransportErrorClassifier,
)

GenerationStreamEventType = Literal["delta", "done"]


@dataclass(frozen=True)
class GenerationStreamEvent:
    type: GenerationStreamEventType
    text: str = ""
    result: GenerationResult | None = None


class GenerationProvider(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

    def stream_generate(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamEvent]:
        ...


class _ResponseStreamContext(Protocol):
    def __enter__(self) -> object:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
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

    def stream_generate(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamEvent]:
        yield GenerationStreamEvent(type="done", result=self.generate(request))


class OpenAIResponsesGenerationProvider:
    provider_name = "openai_responses"
    max_attempts = 3
    retry_delay_seconds = 0.5

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: int = 60,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        http_client: HttpPostClient | HttpStreamClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.http_client = http_client

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

    def stream_generate(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamEvent]:
        stream_method = getattr(self.http_client, "stream", None)
        if not callable(stream_method):
            yield GenerationStreamEvent(type="done", result=self.generate(request))
            return

        body = self._build_request_body(request, stream=True)
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                stream_context = cast(
                    _ResponseStreamContext,
                    stream_method(
                        "POST",
                        f"{self.base_url}/responses",
                        headers=self._headers(),
                        json=body,
                        timeout=self.timeout_seconds,
                    ),
                )
                with stream_context as response:
                    raise_for_status = getattr(response, "raise_for_status", None)
                    if callable(raise_for_status):
                        raise_for_status()
                    yield from self._stream_events_from_response(response)
                    return
            except Exception as exc:  # pragma: no cover - wrapped consistently
                last_error = exc
                if (
                    isinstance(exc, RuntimeError)
                    and str(exc).startswith("Generation provider failed:")
                    and not self._should_retry(exc)
                ):
                    raise
                if attempt >= self.max_attempts or not self._should_retry(exc):
                    raise self._format_retry_failure(attempt, exc) from exc
                time.sleep(self.retry_delay_seconds * attempt)

        assert last_error is not None
        raise self._format_retry_failure(self.max_attempts, last_error) from last_error

    def _sanitize_error_message(self, message: str) -> str:
        if self.api_key:
            return message.replace(self.api_key, "<hidden>")
        return message

    def _wrap_provider_error(self, message: str) -> RuntimeError:
        return RuntimeError(
            f"Generation provider failed: {self._sanitize_error_message(message)}"
        )

    def _format_retry_failure(self, attempt: int, exc: Exception) -> RuntimeError:
        sanitized_message = self._sanitize_error_message(str(exc))
        retries = max(0, attempt - 1)
        if retries == 0:
            return RuntimeError(f"Generation provider failed: {sanitized_message}")
        return RuntimeError(
            "Generation provider failed after "
            f"{attempt} attempts (retried {retries} times): {sanitized_message}"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(
        self,
        request: GenerationRequest,
        *,
        stream: bool = False,
    ) -> dict[str, object]:
        body: dict[str, object] = {
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
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if stream:
            body["stream"] = True
        return body

    def _post(self, request: GenerationRequest) -> object:
        body = self._build_request_body(request)
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                post_method = getattr(self.http_client, "post", None)
                if not callable(post_method):
                    raise RuntimeError("HTTP client does not support POST")
                response = post_method(
                    f"{self.base_url}/responses",
                    headers=self._headers(),
                    json=body,
                    timeout=self.timeout_seconds,
                )
                raise_for_status = getattr(response, "raise_for_status", None)
                if callable(raise_for_status):
                    raise_for_status()
                return response
            except Exception as exc:  # pragma: no cover - wrapped consistently
                last_error = exc
                if attempt >= self.max_attempts or not self._should_retry(exc):
                    raise self._format_retry_failure(attempt, exc) from exc
                time.sleep(self.retry_delay_seconds * attempt)

        assert last_error is not None
        raise self._format_retry_failure(self.max_attempts, last_error) from last_error

    def _stream_events_from_response(
        self,
        response: object,
    ) -> Iterator[GenerationStreamEvent]:
        if not self._is_event_stream_response(response):
            payload = self._read_payload(response)
            answer = self._parse_response_text(payload)
            yield GenerationStreamEvent(
                type="done",
                result=self._success_result(answer, stream=True),
            )
            return

        done_text: str | None = None
        deltas: list[str] = []
        fallback_texts: list[str] = []
        for event_type, data in self._iter_event_stream_response_data(response):
            if data == "[DONE]":
                continue
            payload = self._parse_event_stream_payload(data)
            current_event_type = self._event_type(event_type, payload)
            if current_event_type == "response.output_text.done":
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    done_text = text
                continue
            if current_event_type == "response.output_text.delta":
                delta = payload.get("delta")
                if isinstance(delta, str):
                    deltas.append(delta)
                    yield GenerationStreamEvent(type="delta", text=delta)
                continue

            response_payload = payload.get("response")
            if isinstance(response_payload, dict):
                parsed_response: dict[str, object] = {
                    key: value
                    for key, value in response_payload.items()
                    if isinstance(key, str)
                }
                with suppress(RuntimeError):
                    fallback_texts.append(self._parse_response_text(parsed_response))
                continue

            if "output_text" in payload or "output" in payload:
                with suppress(RuntimeError):
                    fallback_texts.append(self._parse_response_text(payload))

        answer = (done_text or "".join(deltas) or "".join(fallback_texts)).strip()
        if not answer:
            raise self._wrap_provider_error("Could not parse response text")
        yield GenerationStreamEvent(
            type="done",
            result=self._success_result(answer, stream=True),
        )

    def _success_result(self, answer: str, *, stream: bool = False) -> GenerationResult:
        metadata: dict[str, object] = {
            "model": self.model,
            "base_url": self.base_url,
            "endpoint": "/responses",
        }
        if stream:
            metadata["stream"] = True
        return GenerationResult(
            provider_name=self.provider_name,
            status="success",
            answer=answer,
            metadata=metadata,
        )

    def _is_event_stream_response(self, response: object) -> bool:
        headers = getattr(response, "headers", None)
        if isinstance(headers, Mapping):
            content_type = headers.get("content-type") or headers.get("Content-Type")
            if isinstance(content_type, str):
                return "text/event-stream" in content_type.lower()

        response_text = getattr(response, "text", None)
        return isinstance(response_text, str) and response_text.lstrip().startswith(
            "event:"
        )

    def _iter_event_stream_response_data(
        self,
        response: object,
    ) -> Iterator[tuple[str | None, str]]:
        event_type: str | None = None
        data_lines: list[str] = []
        for raw_line in self._iter_response_lines(response):
            line = raw_line.rstrip("\r")
            if not line:
                if data_lines:
                    yield event_type, "\n".join(data_lines)
                event_type = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
        if data_lines:
            yield event_type, "\n".join(data_lines)

    def _iter_response_lines(self, response: object) -> Iterator[str]:
        iter_lines = getattr(response, "iter_lines", None)
        if callable(iter_lines):
            raw_lines = cast(Iterable[object], iter_lines())
            for raw_line in raw_lines:
                if isinstance(raw_line, bytes):
                    yield raw_line.decode("utf-8", errors="replace")
                elif isinstance(raw_line, str):
                    yield raw_line
                else:
                    yield str(raw_line)
            return

        response_text = getattr(response, "text", None)
        if isinstance(response_text, str):
            yield from response_text.splitlines()
            return
        raise self._wrap_provider_error("Could not parse response text")

    def _parse_event_stream_payload(self, data: str) -> dict[str, object]:
        try:
            parsed: object = json.loads(data)
        except json.JSONDecodeError as exc:
            raise self._wrap_provider_error(
                f"Could not parse event stream response: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            return {}
        return {key: value for key, value in parsed.items() if isinstance(key, str)}

    def _event_type(
        self,
        event_type: str | None,
        payload: dict[str, object],
    ) -> str | None:
        payload_type = payload.get("type")
        if isinstance(payload_type, str):
            return payload_type
        return event_type

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(
            self.http_client, HttpTransportErrorClassifier
        ) and self.http_client.is_transport_error(exc):
            return True

        message = str(exc)
        if "HTTP 429" in message or "HTTP 408" in message:
            return True
        for code in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504"):
            if code in message:
                return True
        if "UNEXPECTED_EOF_WHILE_READING" in message:
            return True
        return "EOF occurred in violation of protocol" in message

    def _read_payload(self, response: object) -> dict[str, object]:
        try:
            payload = response.json()  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - wrapped consistently
            raise self._wrap_provider_error(str(exc)) from exc
        if not isinstance(payload, dict):
            raise self._wrap_provider_error("Could not parse response text")
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
        raise self._wrap_provider_error("Could not parse response text")


class GenerationService:
    def __init__(self, provider: GenerationProvider | None = None) -> None:
        self.provider = provider or NullGenerationProvider()

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        http_client: HttpPostClient | HttpStreamClient | None = None,
    ) -> GenerationService:
        generation = config.generation
        provider_name = generation.provider
        if provider_name == "null":
            return cls(NullGenerationProvider())
        if provider_name == "openai_responses":
            base_url = generation.base_url
            model = generation.model
            api_key = generation.api_key
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
            if not api_key:
                raise ValueError(
                    "Invalid config file: generation.api_key is required "
                    "when generation.provider is openai_responses"
                )
            return cls(
                OpenAIResponsesGenerationProvider(
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    timeout_seconds=generation.timeout_seconds,
                    temperature=generation.temperature,
                    reasoning_effort=generation.reasoning_effort,
                    http_client=http_client,
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

    def stream_generate(
        self,
        context_pack: ContextPack,
    ) -> Iterator[GenerationStreamEvent]:
        request = self.build_request(context_pack)
        yield from self.provider.stream_generate(request)
