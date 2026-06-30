from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragent_forge.app.models import AppConfig
from ragent_forge.app.workspace import LocalWorkspace

DEFAULT_CONFIG_TEXT = """[generation]
provider = "null"
"""


class ConfigService:
    def __init__(self, workspace: LocalWorkspace) -> None:
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
        self.workspace.config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
        return self.workspace.config_path

    def _build_config(self, raw_config: dict[str, Any]) -> AppConfig:
        generation = raw_config.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Invalid config file: generation section must be a table")

        provider = generation.get("provider", "null")
        if not isinstance(provider, str):
            raise ValueError(
                "Invalid config file: generation.provider must be a string"
            )
        if provider != "null":
            raise ValueError(f"Unsupported generation provider: {provider}")

        try:
            return AppConfig.model_validate(raw_config)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid config file {self.workspace.config_path}: {exc}"
            ) from exc
