from __future__ import annotations

import time
from typing import Any

from ragent_forge.app.models import AppConfig
from ragent_forge.app.services.eval_dataset_generation_service import SYSTEM_PROMPT
from ragent_forge.app.services.generation_service import (
    OpenAIResponsesGenerationProvider,
)


class OpenAIResponsesTextGenerationClient(OpenAIResponsesGenerationProvider):
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        response = self._post_text(prompt, system_prompt or SYSTEM_PROMPT)
        payload = self._read_payload(response)
        return self._parse_response_text(payload)

    @classmethod
    def from_config(cls, config: AppConfig) -> OpenAIResponsesTextGenerationClient:
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
                last_error = exc
                if attempt >= self.max_attempts or not self._should_retry(exc):
                    raise self._format_retry_failure(attempt, exc) from exc
                time.sleep(self.retry_delay_seconds * attempt)

        assert last_error is not None
        raise self._format_retry_failure(self.max_attempts, last_error) from last_error
