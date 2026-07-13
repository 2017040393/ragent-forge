from __future__ import annotations

from rich.console import Console

from ragent_forge.app.services.trace_history_service import TraceHistoryService
from ragent_forge.infrastructure.local_workspace import LocalWorkspace


def _handle_traces_latest(console: Console, workspace_path: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_latest_trace():
        console.print("No trace found. Run ragent ingest <path> first.")
        return 0
    try:
        trace = workspace.read_latest_trace()
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Traces failed:[/bold red] {exc}")
        return 1
    _print_trace(console, trace)
    return 0


def _handle_traces_list(console: Console, workspace_path: str, limit: int) -> int:
    workspace = LocalWorkspace(workspace_path)
    result = TraceHistoryService(workspace).list_traces(limit)
    if result.traces:
        console.print("Traces")
        console.print("Trace ID | Operation | Status | Started at | Finished at")
        for trace in result.traces:
            console.print(
                f"{trace.trace_id} | {trace.operation} | {trace.status} | "
                f"{trace.started_at} | {trace.finished_at or ''}",
                soft_wrap=True,
            )
    elif result.warnings:
        console.print("No valid traces found.")
    else:
        console.print("No traces found. Run ragent ingest <path> first.")
    _print_trace_warnings(console, result.warnings)
    return 0


def _handle_traces_show(console: Console, workspace_path: str, trace_id: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    try:
        trace = TraceHistoryService(workspace).read_trace(trace_id)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Traces failed:[/bold red] {exc}")
        return 1
    if trace is None:
        console.print(f"Trace not found: {trace_id}")
        return 0
    _print_trace(console, trace)
    return 0


def _print_trace(console: Console, trace: dict[str, object]) -> None:
    console.print(f"Trace ID: {trace.get('trace_id', '')}", soft_wrap=True)
    console.print(f"Operation: {trace.get('operation', '')}")
    console.print(f"Status: {trace.get('status', '')}")
    console.print(f"Started at: {trace.get('started_at', '')}")
    console.print(f"Finished at: {trace.get('finished_at', '')}")
    console.print()
    console.print("Steps:")
    steps = trace.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    for index, step in enumerate(steps, start=1):
        step_name = step.get("name", "") if isinstance(step, dict) else ""
        console.print(f"{index}. {step_name}")
    console.print()
    console.print("Metadata:")
    metadata = trace.get("metadata", {})
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            console.print(f"- {key}: {value}", soft_wrap=True)


def _print_trace_warnings(console: Console, warnings: list[str]) -> None:
    if not warnings:
        return
    console.print()
    console.print("Warnings:")
    for warning in warnings:
        console.print(f"- {warning}", soft_wrap=True)
