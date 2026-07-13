from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from ragent_forge.app.models import AppConfig
from ragent_forge.app.ports import HttpPostClient, HttpStreamClient
from ragent_forge.app.services.eval_dataset_generation_service import SYSTEM_PROMPT
from ragent_forge.app.services.generation_service import (
    OpenAIResponsesGenerationProvider,
)


class OpenAIResponsesTextGenerationClient(OpenAIResponsesGenerationProvider):
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        response = self._post_text(prompt, system_prompt or SYSTEM_PROMPT)
        if self._is_event_stream_response(response):
            return self._parse_event_stream_response_text(response)
        payload = self._read_payload(response)
        return self._parse_response_text(payload)

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

    def _parse_event_stream_response_text(self, response: object) -> str:
        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str):
            raise self._wrap_provider_error("Could not parse response text")

        done_text: str | None = None
        deltas: list[str] = []
        fallback_texts: list[str] = []
        for event_type, data in self._iter_event_stream_data(response_text):
            if data == "[DONE]":
                continue
            try:
                parsed: object = json.loads(data)
            except json.JSONDecodeError as exc:
                raise self._wrap_provider_error(
                    f"Could not parse event stream response: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                continue

            payload: dict[str, object] = {}
            for key, value in parsed.items():
                if isinstance(key, str):
                    payload[key] = value

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
                continue
            if "output_text" in payload or "output" in payload:
                try:
                    fallback_texts.append(self._parse_response_text(payload))
                except RuntimeError:
                    continue

        answer = (done_text or "".join(deltas) or "".join(fallback_texts)).strip()
        if answer:
            return answer
        raise self._wrap_provider_error("Could not parse response text")

    def _event_type(
        self, event_type: str | None, payload: dict[str, object]
    ) -> str | None:
        payload_type = payload.get("type")
        if isinstance(payload_type, str):
            return payload_type
        return event_type

    def _iter_event_stream_data(self, stream_text: str) -> list[tuple[str | None, str]]:
        events: list[tuple[str | None, str]] = []
        event_type: str | None = None
        data_lines: list[str] = []
        for raw_line in stream_text.splitlines():
            line = raw_line.rstrip("\r")
            if not line:
                if data_lines:
                    events.append((event_type, "\n".join(data_lines)))
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
            events.append((event_type, "\n".join(data_lines)))
        return events

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        http_client: HttpPostClient | HttpStreamClient | None = None,
    ) -> OpenAIResponsesTextGenerationClient:
        generation = config.generation
        if generation.provider != "openai_responses":
            raise ValueError(f"Unsupported generation provider: {generation.provider}")

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
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=generation.timeout_seconds,
            temperature=generation.temperature,
            reasoning_effort=generation.reasoning_effort,
            http_client=http_client,
        )

    def _post_text(self, prompt: str, system_prompt: str) -> object:
        body: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
        }
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                post_method = getattr(self.http_client, "post", None)
                if not callable(post_method):
                    raise RuntimeError("HTTP client does not support POST")
                response = post_method(
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
                last_error = exc
                if attempt >= self.max_attempts or not self._should_retry(exc):
                    raise self._format_retry_failure(attempt, exc) from exc
                time.sleep(self.retry_delay_seconds * attempt)

        assert last_error is not None
        raise self._format_retry_failure(self.max_attempts, last_error) from last_error
