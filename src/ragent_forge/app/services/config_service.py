from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragent_forge.app.models import AppConfig
from ragent_forge.app.ports import ConfigWorkspace
from ragent_forge.app.storage import atomic_write_text

DEFAULT_CONFIG_TEXT = """[generation]
provider = "null"

[embedding]
provider = "none"
"""

SUPPORTED_GENERATION_KEYS = {
    "provider",
    "base_url",
    "model",
    "api_key",
    "timeout_seconds",
    "temperature",
    "reasoning_effort",
}

SUPPORTED_EMBEDDING_KEYS = {
    "provider",
    "base_url",
    "model",
    "api_key",
    "timeout_seconds",
    "batch_size",
}


class ConfigService:
    def __init__(self, workspace: ConfigWorkspace) -> None:
        self.workspace = workspace

    def default_config(self) -> AppConfig:
        return AppConfig()

    def load(self) -> AppConfig:
        if not self.workspace.config_path.is_file():
            return self.default_config()

        try:
            raw_config = tomllib.loads(
                self.workspace.config_path.read_text(encoding="utf-8")
            )
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"Invalid TOML in config file {self.workspace.config_path}: {exc}"
            ) from exc

        return self._build_config(raw_config)

    def write_default(self, overwrite: bool = False) -> Path:
        if self.workspace.config_path.exists() and not overwrite:
            return self.workspace.config_path

        self.workspace.root_path.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.workspace.config_path, DEFAULT_CONFIG_TEXT)
        return self.workspace.config_path

    def _build_config(self, raw_config: dict[str, Any]) -> AppConfig:
        generation = raw_config.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Invalid config file: generation section must be a table")
        embedding = raw_config.get("embedding", {})
        if not isinstance(embedding, dict):
            raise ValueError("Invalid config file: embedding section must be a table")

        provider = generation.get("provider", "null")
        if not isinstance(provider, str):
            raise ValueError(
                "Invalid config file: generation.provider must be a string"
            )
        if provider not in {"null", "openai_responses"}:
            raise ValueError(f"Unsupported generation provider: {provider}")
        unknown_generation_keys = sorted(
            key for key in generation if key not in SUPPORTED_GENERATION_KEYS
        )
        if unknown_generation_keys:
            formatted_keys = ", ".join(
                f"generation.{key}" for key in unknown_generation_keys
            )
            raise ValueError(
                "Invalid config file: unsupported generation settings: "
                f"{formatted_keys}"
            )
        if provider == "openai_responses":
            base_url = generation.get("base_url")
            model = generation.get("model")
            api_key = generation.get("api_key")
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError(
                    "Invalid config file: generation.base_url is required "
                    "when generation.provider is openai_responses"
                )
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "Invalid config file: generation.model is required "
                    "when generation.provider is openai_responses"
                )
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError(
                    "Invalid config file: generation.api_key is required "
                    "when generation.provider is openai_responses"
                )

        embedding_provider = embedding.get("provider", "none")
        if not isinstance(embedding_provider, str):
            raise ValueError("Invalid config file: embedding.provider must be a string")
        if embedding_provider not in {"none", "openai_embeddings"}:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
        unknown_embedding_keys = sorted(
            key for key in embedding if key not in SUPPORTED_EMBEDDING_KEYS
        )
        if unknown_embedding_keys:
            formatted_keys = ", ".join(
                f"embedding.{key}" for key in unknown_embedding_keys
            )
            raise ValueError(
                "Invalid config file: unsupported embedding settings: "
                f"{formatted_keys}"
            )
        if embedding_provider == "openai_embeddings":
            base_url = embedding.get("base_url")
            model = embedding.get("model")
            api_key = embedding.get("api_key")
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError(
                    "Invalid config file: embedding.base_url is required "
                    "when embedding.provider is openai_embeddings"
                )
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "Invalid config file: embedding.model is required "
                    "when embedding.provider is openai_embeddings"
                )
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError(
                    "Invalid config file: embedding.api_key is required "
                    "when embedding.provider is openai_embeddings"
                )

        try:
            return AppConfig.model_validate(raw_config)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid config file {self.workspace.config_path}: {exc}"
            ) from exc
