from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from ragent_forge.app.models import WorkspaceStatus
from ragent_forge.app.services.ingest_service import IngestService
from ragent_forge.app.services.trace_service import build_ingest_trace
from ragent_forge.infrastructure.local_workspace import LocalWorkspace


def handle_workspace_migrate(
    console: Console,
    workspace_path: str,
    *,
    dry_run: bool,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    try:
        migration = workspace.migrate_legacy_workspace(dry_run=dry_run)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Workspace migration failed:[/bold red] {exc}")
        return 1

    title = (
        "Workspace migration dry run"
        if migration.dry_run
        else "Workspace migration complete"
    )
    console.print(f"[bold green]{title}[/bold green]")
    console.print(f"Source layout: {migration.source_layout}")
    console.print(f"Migration required: {str(migration.required).lower()}")
    if migration.snapshot_id is not None:
        console.print(f"Workspace snapshot: {migration.snapshot_id}")
    for action in migration.actions:
        console.print(f"- {action}")
    return 0


def handle_ingest(
    console: Console,
    source_path: str,
    workspace_path: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    started_at = datetime.now(UTC)
    try:
        result = IngestService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ).ingest(source_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Ingest failed:[/bold red] {exc}")
        return 1

    workspace = LocalWorkspace(workspace_path)
    try:
        generation = workspace.commit_ingest_generation(result)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Ingest commit failed:[/bold red] {exc}")
        return 1

    snapshot_id = generation.snapshot_id
    chunks_path = Path(generation.chunks_path)
    summary_path = Path(generation.ingest_summary_path)
    trace = build_ingest_trace(
        result=result,
        chunks_path=chunks_path,
        summary_path=summary_path,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        snapshot_id=snapshot_id,
    )
    trace_path = workspace.write_trace(trace, snapshot_id)
    console.print("[bold green]Ingest complete[/bold green]")
    console.print(f"Source: [cyan]{result.source_path}[/cyan]")
    console.print(f"Documents: {result.document_count}")
    console.print(f"Chunks: {result.chunk_count}")
    console.print(f"Skipped files: {result.skipped_count}")
    console.print(f"Chunk size: {result.metadata['chunk_size']}")
    console.print(f"Chunk overlap: {result.metadata['chunk_overlap']}")
    console.print(f"Saved chunks to: {chunks_path}")
    console.print(f"Saved summary to: {summary_path}")
    console.print(f"Saved trace to: {trace_path}")
    console.print(f"Workspace snapshot: {snapshot_id}")
    return 0


def handle_status(console: Console, workspace_path: str) -> int:
    try:
        workspace_status = LocalWorkspace(workspace_path).status()
    except ValueError as exc:
        console.print(f"[bold red]Status failed:[/bold red] {exc}")
        return 1
    _print_workspace_status(console, workspace_status)
    return 0


def _print_workspace_status(
    console: Console, workspace_status: WorkspaceStatus
) -> None:
    console.print(f"Workspace: [cyan]{workspace_status.root_path}[/cyan]")
    if workspace_status.status == "not_initialized":
        console.print("Status: not initialized")
        console.print()
        console.print("Run `ragent ingest <path>` to create a local workspace.")
        return
    if workspace_status.status == "incomplete":
        console.print("Status: incomplete")
        console.print()
        for missing_file in workspace_status.missing_files:
            if missing_file == workspace_status.latest_summary_path:
                console.print(f"Missing summary file: {missing_file}")
            elif missing_file == workspace_status.chunks_path:
                console.print(f"Missing chunks file: {missing_file}")
            else:
                console.print(f"Missing file: {missing_file}")
        console.print("Run `ragent ingest <path>` to regenerate workspace data.")
        return
    summary = workspace_status.summary
    console.print("Status: ready")
    console.print()
    console.print(f"Last ingest source: {summary.get('source_path', '')}")
    console.print(f"Documents: {summary.get('document_count', 0)}")
    chunk_count = (
        workspace_status.chunk_count_from_file
        if workspace_status.chunk_count_from_file is not None
        else summary.get("chunk_count", 0)
    )
    console.print(f"Chunks: {chunk_count}")
    console.print(f"Skipped files: {summary.get('skipped_count', 0)}")
    console.print()
    console.print(f"Chunks file: {workspace_status.chunks_path}")
    console.print(f"Summary file: {workspace_status.latest_summary_path}")
