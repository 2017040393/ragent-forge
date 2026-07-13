from __future__ import annotations

from rich.console import Console

from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.infrastructure.local_workspace import LocalWorkspace


def _handle_config_show(console: Console, workspace_path: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    config_service = ConfigService(workspace)
    try:
        config = config_service.load()
    except ValueError as exc:
        console.print(f"[bold red]Config failed:[/bold red] {exc}")
        return 1
    if workspace.config_path.is_file():
        console.print(f"Config: {workspace.config_path}", soft_wrap=True)
    else:
        console.print("Config: default")
    console.print()
    console.print(f"generation.provider: {config.generation.provider}")
    if config.generation.provider == "openai_responses":
        console.print(f"generation.base_url: {config.generation.base_url}")
        console.print(f"generation.model: {config.generation.model}")
        if config.generation.api_key:
            console.print("generation.api_key: <hidden>")
        console.print(
            f"generation.timeout_seconds: {config.generation.timeout_seconds}"
        )
        console.print(f"generation.temperature: {config.generation.temperature}")
        if config.generation.reasoning_effort is not None:
            console.print(
                f"generation.reasoning_effort: {config.generation.reasoning_effort}"
            )
    console.print(f"embedding.provider: {config.embedding.provider}")
    if config.embedding.provider == "openai_embeddings":
        console.print(f"embedding.base_url: {config.embedding.base_url}")
        console.print(f"embedding.model: {config.embedding.model}")
        if config.embedding.api_key:
            console.print("embedding.api_key: <hidden>")
        console.print(f"embedding.timeout_seconds: {config.embedding.timeout_seconds}")
        console.print(f"embedding.batch_size: {config.embedding.batch_size}")
    return 0


def _handle_config_init(console: Console, workspace_path: str, overwrite: bool) -> int:
    workspace = LocalWorkspace(workspace_path)
    config_service = ConfigService(workspace)
    config_exists = workspace.config_path.exists()
    config_path = config_service.write_default(overwrite=overwrite)
    if config_exists and (not overwrite):
        console.print(f"Config already exists: {config_path}", soft_wrap=True)
        return 0
    console.print(f"Wrote default config to: {config_path}", soft_wrap=True)
    return 0
