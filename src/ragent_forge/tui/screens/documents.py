from pathlib import Path

from textual.widgets import Static

from ragent_forge.app.models import WorkspaceStatus
from ragent_forge.app.services.chunk_service import ChunkService, make_preview
from ragent_forge.app.workspace import LocalWorkspace


def load_workspace_status_text(workspace_path: str | Path = ".ragent") -> str:
    workspace = LocalWorkspace(workspace_path)
    try:
        status = workspace.status()
    except ValueError as exc:
        return f"Workspace status error:\n{exc}"
    return format_workspace_status(status, workspace)


def format_workspace_status(
    status: WorkspaceStatus,
    workspace: LocalWorkspace | None = None,
) -> str:
    lines = [f"Workspace: {status.root_path}"]

    if status.status == "not_initialized":
        lines.extend(
            [
                "Status: not initialized",
                "",
                "Run `ragent ingest <path>` to create a local workspace.",
            ]
        )
        return "\n".join(lines)

    if status.status == "incomplete":
        lines.extend(["Status: incomplete", "", "Missing files:"])
        lines.extend(f"- {missing_file}" for missing_file in status.missing_files)
        lines.extend(
            [
                "",
                "Run `ragent ingest <path>` to regenerate workspace data.",
            ]
        )
        return "\n".join(lines)

    summary = status.summary
    chunk_count = (
        status.chunk_count_from_file
        if status.chunk_count_from_file is not None
        else summary.get("chunk_count", 0)
    )
    lines.extend(
        [
            "Status: ready",
            "",
            f"Last ingest source: {summary.get('source_path', '')}",
            f"Documents: {summary.get('document_count', 0)}",
            f"Chunks: {chunk_count}",
            f"Skipped files: {summary.get('skipped_count', 0)}",
            "",
            f"Chunks file: {status.chunks_path}",
            f"Summary file: {status.latest_summary_path}",
        ]
    )
    if workspace is not None:
        lines.extend(format_recent_chunks(workspace))
    return "\n".join(lines)


def format_recent_chunks(workspace: LocalWorkspace, limit: int = 5) -> list[str]:
    try:
        chunks = ChunkService(workspace).list_chunks(limit)
    except (OSError, ValueError):
        return []

    if not chunks:
        return []

    lines = ["", "Recent chunks:"]
    for chunk in chunks:
        lines.append(
            "- "
            f"{chunk.get('chunk_id', '')} | "
            f"{chunk.get('source_path', '')} | "
            f"{chunk.get('start_char', '')}-{chunk.get('end_char', '')} | "
            f"{make_preview(str(chunk.get('text', '')))}"
        )
    return lines


class DocumentsScreen(Static):
    DEFAULT_CSS = "DocumentsScreen { padding: 1; }"

    def __init__(self) -> None:
        super().__init__(load_workspace_status_text())
