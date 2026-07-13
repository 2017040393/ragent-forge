from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragent_forge.app.models import TraceListItem, TraceListResult
from ragent_forge.app.ports import TraceWorkspace


class TraceHistoryService:
    def __init__(self, workspace: TraceWorkspace) -> None:
        self.workspace = workspace

    def list_traces(self, limit: int = 20) -> TraceListResult:
        if not self.workspace.traces_dir.is_dir():
            return TraceListResult()

        traces: list[TraceListItem] = []
        warnings: list[str] = []
        for path in self._trace_files():
            try:
                trace = self._read_trace_file(path)
            except ValueError as exc:
                warnings.append(f"Skipped invalid trace file: {path}: {exc}")
                continue
            traces.append(self._trace_list_item(trace, path))

        traces.sort(key=lambda trace: (trace.started_at, trace.trace_id), reverse=True)
        return TraceListResult(traces=traces[: max(limit, 0)], warnings=warnings)

    def read_trace(self, trace_id: str) -> dict[str, Any] | None:
        if not self._is_safe_trace_id(trace_id):
            return None

        path = self.workspace.traces_dir / f"{trace_id}.json"
        if not path.is_file():
            return None
        return self._read_trace_file(path)

    def _trace_files(self) -> list[Path]:
        return [
            path
            for path in self.workspace.traces_dir.glob("*.json")
            if path.name != "latest_trace.json"
        ]

    def _read_trace_file(self, path: Path) -> dict[str, Any]:
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in trace file {path}: {exc.msg}") from exc
        if not isinstance(trace, dict):
            raise ValueError(f"Invalid JSON in trace file {path}: expected object")
        return trace

    def _trace_list_item(self, trace: dict[str, Any], path: Path) -> TraceListItem:
        return TraceListItem(
            trace_id=str(trace.get("trace_id", path.stem)),
            operation=str(trace.get("operation", "")),
            status=str(trace.get("status", "")),
            started_at=str(trace.get("started_at", "")),
            finished_at=self._optional_string(trace.get("finished_at")),
            path=str(path),
        )

    def _optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _is_safe_trace_id(self, trace_id: str) -> bool:
        return trace_id == Path(trace_id).name and "\\" not in trace_id
