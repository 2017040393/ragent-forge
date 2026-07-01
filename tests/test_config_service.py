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


def test_loading_openai_responses_config_returns_generation_provider(
    tmp_path: Path,
) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        (
            "[generation]\n"
            "provider = \"openai_responses\"\n"
            "base_url = \"https://api.openai.com/v1\"\n"
            "model = \"gpt-4o-mini\"\n"
            "api_key = \"super-secret-key\"\n"
            "timeout_seconds = 60\n"
            "temperature = 0.2\n"
            "reasoning_effort = \"low\"\n"
        ),
        encoding="utf-8",
    )

    config = service.load()

    assert config.generation.provider == "openai_responses"
    assert config.generation.base_url == "https://api.openai.com/v1"
    assert config.generation.model == "gpt-4o-mini"
    assert config.generation.api_key == "super-secret-key"
    assert config.generation.timeout_seconds == 60
    assert config.generation.temperature == 0.2
    assert config.generation.reasoning_effort == "low"


def test_loading_openai_responses_config_requires_base_url(tmp_path: Path) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = "super-secret-key"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid config file: generation.base_url is required "
            "when generation.provider is openai_responses"
        ),
    ):
        service.load()


def test_loading_openai_responses_config_requires_model(tmp_path: Path) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'api_key = "super-secret-key"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid config file: generation.model is required "
            "when generation.provider is openai_responses"
        ),
    ):
        service.load()


def test_loading_openai_responses_config_requires_api_key_env(
    tmp_path: Path,
) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "gpt-4o-mini"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid config file: generation.api_key is required "
            "when generation.provider is openai_responses"
        ),
    ):
        service.load()


def test_loading_openai_responses_config_rejects_legacy_api_key_env(
    tmp_path: Path,
) -> None:
    service = make_config_service(tmp_path)
    service.workspace.root_path.mkdir(parents=True)
    service.workspace.config_path.write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key_env = "OPENAI_API_KEY"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid config file: generation.api_key_env is no longer supported; "
            "use generation.api_key instead"
        ),
    ):
        service.load()


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
