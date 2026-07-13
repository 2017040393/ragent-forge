from __future__ import annotations

import json
from collections.abc import Mapping

from rich.console import Console

from ragent_forge.app.services.chunk_service import ChunkService, make_preview
from ragent_forge.app.source_labels import format_source_label, format_source_range
from ragent_forge.infrastructure.local_workspace import LocalWorkspace


def _handle_chunks_list(console: Console, workspace_path: str, limit: int) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0
    service = ChunkService(workspace)
    try:
        chunks = service.list_chunks(limit)
        total_count = service.count_chunks()
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Chunks failed:[/bold red] {exc}")
        return 1
    console.print("Chunks")
    console.print("Chunk ID | Source | Range | Preview")
    for chunk in chunks:
        metadata = chunk.get("metadata")
        source_label = format_source_label(
            str(chunk.get("source_path", "")),
            metadata if isinstance(metadata, dict) else None,
        )
        console.print(
            f"{chunk.get('chunk_id', '')} | {source_label} | "
            f"{_format_char_range(chunk)} | "
            f"{make_preview(str(chunk.get('text', '')))}",
            soft_wrap=True,
        )
    if total_count > limit:
        console.print(
            f"Showing {len(chunks)} of {total_count} chunks. Use --limit to show more."
        )
    return 0


def _handle_chunks_show(console: Console, workspace_path: str, chunk_id: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0
    try:
        chunk = ChunkService(workspace).get_chunk(chunk_id)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Chunks failed:[/bold red] {exc}")
        return 1
    if chunk is None:
        console.print(f"Chunk not found: {chunk_id}")
        return 0
    console.print(f"[bold]Chunk ID:[/bold] {chunk.get('chunk_id', '')}", soft_wrap=True)
    console.print(
        f"[bold]Document ID:[/bold] {chunk.get('document_id', '')}", soft_wrap=True
    )
    console.print(
        f"[bold]Source path:[/bold] {chunk.get('source_path', '')}", soft_wrap=True
    )
    console.print(f"[bold]Start char:[/bold] {chunk.get('start_char', '')}")
    console.print(f"[bold]End char:[/bold] {chunk.get('end_char', '')}")
    console.print()
    console.print("[bold]Metadata:[/bold]")
    console.print(json.dumps(chunk.get("metadata", {}), ensure_ascii=False, indent=2))
    console.print()
    console.print("[bold]Text:[/bold]")
    console.print(str(chunk.get("text", "")))
    return 0


def _format_char_range(chunk: Mapping[str, object]) -> str:
    start_char = chunk.get("start_char")
    end_char = chunk.get("end_char")
    metadata = chunk.get("metadata")
    return format_source_range(
        start_char if isinstance(start_char, int) else None,
        end_char if isinstance(end_char, int) else None,
        metadata if isinstance(metadata, dict) else None,
    )


def _print_no_chunks(console: Console) -> None:
    console.print("No chunks found. Run ragent ingest <path> first.")
