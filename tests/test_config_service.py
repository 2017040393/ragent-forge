from pathlib import Path

import pytest

from ragent_forge.app.services.config_service import (
    DEFAULT_CONFIG_TEXT,
    ConfigService,
)
from ragent_forge.app.workspace import LocalWorkspace


def make_config_service(tmp_path: Path) -> ConfigService:
    return ConfigService(LocalWorkspace(tmp_path / ".ragent"))


def test_missing_config_returns_default_generation_provider(tmp_path: Path) -> None:
    config = make_config_service(tmp_path).load()

    assert config.generation.provider == "null"


def test_write_default_creates_config_file(tmp_path: Path) -> None:
    service = make_config_service(tmp_path)

    config_path = service.write_default()

    assert config_path == service.workspace.config_path
    assert config_path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEXT


def test_write_default_does_not_overwrite_existing_config(tmp_path: Path) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        "[generation]\nprovider = \"custom\"\n",
        encoding="utf-8",
    )

    config_path = service.write_default()

    assert config_path == service.workspace.config_path
    assert service.workspace.config_path.read_text(encoding="utf-8") == (
        "[generation]\nprovider = \"custom\"\n"
    )


def test_write_default_overwrites_existing_config_when_requested(
    tmp_path: Path,
) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        "[generation]\nprovider = \"custom\"\n",
        encoding="utf-8",
    )

    service.write_default(overwrite=True)

    assert service.workspace.config_path.read_text(encoding="utf-8") == (
        DEFAULT_CONFIG_TEXT
    )


def test_loading_valid_config_returns_generation_provider(tmp_path: Path) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        "[generation]\nprovider = \"null\"\n",
        encoding="utf-8",
    )

    config = service.load()

    assert config.generation.provider == "null"


def test_loading_invalid_toml_raises_clear_value_error(tmp_path: Path) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text("[generation\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid TOML in config file"):
        service.load()


def test_loading_unsupported_provider_raises_clear_value_error(
    tmp_path: Path,
) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        "[generation]\nprovider = \"openai\"\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported generation provider: openai"):
        service.load()
