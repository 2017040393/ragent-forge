from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.widgets import Static

from ragent_forge.app.models import TraceListResult
from ragent_forge.app.services.trace_history_service import TraceHistoryService
from ragent_forge.infrastructure.local_workspace import LocalWorkspace


def load_latest_trace_text(workspace_path: str | Path = ".ragent") -> str:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_latest_trace():
        return "\n".join(
            [
                "Latest trace",
                "",
                "No trace found. Run `ragent ingest <path>` first.",
            ]
        )

    try:
        trace = workspace.read_latest_trace()
    except ValueError as exc:
        return "\n".join(["Latest trace", "", "Trace status error:", str(exc)])

    return format_latest_trace(trace)


def load_trace_view_text(workspace_path: str | Path = ".ragent") -> str:
    workspace = LocalWorkspace(workspace_path)
    history = TraceHistoryService(workspace).list_traces(limit=5)
    return "\n".join(
        [
            load_latest_trace_text(workspace_path),
            "",
            format_trace_history(history),
        ]
    )


def format_latest_trace(trace: dict[str, Any]) -> str:
    lines = [
        "Latest trace",
        "",
        f"Trace ID: {trace.get('trace_id', '')}",
        f"Operation: {trace.get('operation', '')}",
        f"Status: {trace.get('status', '')}",
        f"Started at: {trace.get('started_at', '')}",
        f"Finished at: {trace.get('finished_at', '')}",
        "",
        "Steps:",
    ]

    for index, step in enumerate(trace.get("steps", []), start=1):
        step_name = step.get("name", "") if isinstance(step, dict) else ""
        lines.append(f"{index}. {step_name}")

    lines.extend(["", "Metadata:"])
    metadata = trace.get("metadata", {})
    if isinstance(metadata, dict):
        lines.extend(f"- {key}: {value}" for key, value in metadata.items())

    return "\n".join(lines)


def format_trace_history(result: TraceListResult) -> str:
    lines = ["Recent traces", ""]

    if result.traces:
        lines.append("Trace ID | Operation | Status | Started at")
        lines.extend(
            (
                f"{trace.trace_id} | {trace.operation} | "
                f"{trace.status} | {trace.started_at}"
            )
            for trace in result.traces
        )
    else:
        lines.append("No traces found. Run `ragent ingest <path>` first.")

    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in result.warnings)

    return "\n".join(lines)


class TraceScreen(Static):
    DEFAULT_CSS = "TraceScreen { padding: 1; }"

    def __init__(self) -> None:
        super().__init__(load_trace_view_text())
