import json
from pathlib import Path
from typing import Any

from ragent_forge.app.services.trace_history_service import TraceHistoryService
from ragent_forge.app.workspace import LocalWorkspace


def write_trace_file(
    traces_dir: Path,
    trace_id: str,
    started_at: str,
    *,
    operation: str = "search",
    finished_at: str | None = "2026-06-30T12:00:01Z",
) -> Path:
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{trace_id}.json"
    path.write_text(
        json.dumps(
            {
                "trace_id": trace_id,
                "operation": operation,
                "status": "success",
                "started_at": started_at,
                "finished_at": finished_at,
                "steps": [],
                "metadata": {"source": "test"},
            }
        ),
        encoding="utf-8",
    )
    return path


def read_trace_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_missing_traces_directory_returns_empty_list_and_no_warnings(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")

    result = TraceHistoryService(workspace).list_traces()

    assert result.traces == []
    assert result.warnings == []


def test_listing_traces_ignores_latest_trace_json(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    path = write_trace_file(
        workspace.traces_dir,
        "search-20260630T120000Z",
        "2026-06-30T12:00:00Z",
    )
    workspace.latest_trace_path.write_text(path.read_text(encoding="utf-8"))

    result = TraceHistoryService(workspace).list_traces()

    assert [trace.trace_id for trace in result.traces] == [
        "search-20260630T120000Z"
    ]
    assert "latest_trace" not in [trace.trace_id for trace in result.traces]


def test_listing_traces_returns_valid_trace_items(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    path = write_trace_file(
        workspace.traces_dir,
        "ingest-20260630T115800Z",
        "2026-06-30T11:58:00Z",
        operation="ingest",
        finished_at="2026-06-30T11:58:02Z",
    )

    result = TraceHistoryService(workspace).list_traces()

    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.trace_id == "ingest-20260630T115800Z"
    assert trace.operation == "ingest"
    assert trace.status == "success"
    assert trace.started_at == "2026-06-30T11:58:00Z"
    assert trace.finished_at == "2026-06-30T11:58:02Z"
    assert trace.path == str(path)


def test_listing_traces_sorts_newest_first_by_started_at(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    write_trace_file(
        workspace.traces_dir,
        "ingest-20260630T115800Z",
        "2026-06-30T11:58:00Z",
        operation="ingest",
    )
    write_trace_file(
        workspace.traces_dir,
        "search-20260630T115900Z",
        "2026-06-30T11:59:00Z",
        operation="search",
    )
    write_trace_file(
        workspace.traces_dir,
        "ask-retrieval-20260630T120003Z",
        "2026-06-30T12:00:03Z",
        operation="ask_retrieval",
    )
    write_trace_file(
        workspace.traces_dir,
        "z-fallback",
        "2026-06-30T11:59:00Z",
        operation="search",
    )

    result = TraceHistoryService(workspace).list_traces()

    assert [trace.trace_id for trace in result.traces] == [
        "ask-retrieval-20260630T120003Z",
        "z-fallback",
        "search-20260630T115900Z",
        "ingest-20260630T115800Z",
    ]


def test_listing_traces_respects_limit(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    write_trace_file(
        workspace.traces_dir,
        "ingest-20260630T115800Z",
        "2026-06-30T11:58:00Z",
    )
    write_trace_file(
        workspace.traces_dir,
        "search-20260630T115900Z",
        "2026-06-30T11:59:00Z",
    )

    result = TraceHistoryService(workspace).list_traces(limit=1)

    assert [trace.trace_id for trace in result.traces] == [
        "search-20260630T115900Z"
    ]


def test_listing_traces_skips_corrupt_json_and_records_warnings(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    write_trace_file(
        workspace.traces_dir,
        "search-20260630T115900Z",
        "2026-06-30T11:59:00Z",
    )
    bad_path = workspace.traces_dir / "bad.json"
    bad_path.write_text("{not-json", encoding="utf-8")

    result = TraceHistoryService(workspace).list_traces()

    assert [trace.trace_id for trace in result.traces] == [
        "search-20260630T115900Z"
    ]
    assert len(result.warnings) == 1
    assert f"Skipped invalid trace file: {bad_path}" in result.warnings[0]
    assert "Invalid JSON" in result.warnings[0]


def test_listing_traces_handles_all_invalid_files_with_warnings(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.traces_dir.mkdir(parents=True)
    (workspace.traces_dir / "bad.json").write_text("{not-json", encoding="utf-8")
    (workspace.traces_dir / "array.json").write_text("[]", encoding="utf-8")

    result = TraceHistoryService(workspace).list_traces()

    assert result.traces == []
    assert len(result.warnings) == 2


def test_read_trace_returns_trace_dict_for_existing_trace(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    path = write_trace_file(
        workspace.traces_dir,
        "search-20260630T115900Z",
        "2026-06-30T11:59:00Z",
    )

    trace = TraceHistoryService(workspace).read_trace("search-20260630T115900Z")

    assert trace == read_trace_json(path)


def test_read_trace_returns_none_for_missing_trace(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")

    trace = TraceHistoryService(workspace).read_trace("missing")

    assert trace is None


def test_read_trace_does_not_read_outside_traces_directory(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    outside_path = workspace.root_path / "outside.json"
    outside_path.parent.mkdir(parents=True)
    outside_path.write_text(
        json.dumps(
            {
                "trace_id": "outside",
                "operation": "search",
                "status": "success",
                "started_at": "2026-06-30T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    trace = TraceHistoryService(workspace).read_trace("../outside")

    assert trace is None
