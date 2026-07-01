from pathlib import Path

from ragent_forge.app.models import (
    OperationTrace,
    TraceListItem,
    TraceListResult,
    TraceStep,
)
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.tui.screens.trace import (
    format_latest_trace,
    format_trace_history,
    load_latest_trace_text,
    load_trace_view_text,
)


def make_trace() -> OperationTrace:
    return OperationTrace(
        trace_id="ingest-20260630T000000Z",
        operation="ingest",
        status="success",
        started_at="2026-06-30T00:00:00Z",
        finished_at="2026-06-30T00:00:02Z",
        steps=[
            TraceStep(name="load_documents", description="Load documents."),
            TraceStep(name="chunk_documents", description="Chunk documents."),
            TraceStep(name="write_chunks", description="Write chunks."),
        ],
        metadata={
            "source_path": "/knowledge",
            "document_count": 2,
            "chunk_count": 8,
            "skipped_count": 0,
        },
    )


def test_format_latest_trace_renders_summary_steps_and_metadata() -> None:
    trace = make_trace().model_dump(mode="json", exclude_none=True)

    text = format_latest_trace(trace)

    assert "Latest trace" in text
    assert "Trace ID: ingest-20260630T000000Z" in text
    assert "Operation: ingest" in text
    assert "Status: success" in text
    assert "Started at: 2026-06-30T00:00:00Z" in text
    assert "Finished at: 2026-06-30T00:00:02Z" in text
    assert "Steps:" in text
    assert "1. load_documents" in text
    assert "2. chunk_documents" in text
    assert "3. write_chunks" in text
    assert "Metadata:" in text
    assert "- source_path: /knowledge" in text
    assert "- document_count: 2" in text
    assert "- chunk_count: 8" in text
    assert "- skipped_count: 0" in text


def test_load_latest_trace_text_returns_friendly_message_when_missing(
    tmp_path: Path,
) -> None:
    text = load_latest_trace_text(tmp_path / ".ragent")

    assert "Latest trace" in text
    assert "No trace found. Run `ragent ingest <path>` first." in text


def test_load_latest_trace_text_renders_trace_from_workspace(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_trace(make_trace())

    text = load_latest_trace_text(workspace.root_path)

    assert "Trace ID: ingest-20260630T000000Z" in text
    assert "1. load_documents" in text
    assert "- chunk_count: 8" in text


def test_load_latest_trace_text_handles_corrupt_latest_trace_json(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    workspace.latest_trace_path.write_text("{not-json", encoding="utf-8")

    text = load_latest_trace_text(workspace.root_path)

    assert "Latest trace" in text
    assert "Trace status error:" in text
    assert "Invalid JSON in latest trace" in text


def test_format_trace_history_renders_valid_traces() -> None:
    result = TraceListResult(
        traces=[
            TraceListItem(
                trace_id="ask-retrieval-20260630T120003Z",
                operation="ask_retrieval",
                status="success",
                started_at="2026-06-30T12:00:03Z",
                finished_at="2026-06-30T12:00:04Z",
                path=".ragent/traces/ask-retrieval-20260630T120003Z.json",
            )
        ]
    )

    text = format_trace_history(result)

    assert "Recent traces" in text
    assert "Trace ID | Operation | Status | Started at" in text
    assert (
        "ask-retrieval-20260630T120003Z | ask_retrieval | success | "
        "2026-06-30T12:00:03Z"
    ) in text


def test_format_trace_history_renders_friendly_message_when_empty() -> None:
    text = format_trace_history(TraceListResult())

    assert "Recent traces" in text
    assert "No traces found. Run `ragent ingest <path>` first." in text


def test_format_trace_history_renders_warnings_after_rows() -> None:
    result = TraceListResult(
        traces=[
            TraceListItem(
                trace_id="search-20260630T115900Z",
                operation="search",
                status="success",
                started_at="2026-06-30T11:59:00Z",
                finished_at=None,
                path=".ragent/traces/search-20260630T115900Z.json",
            )
        ],
        warnings=["Skipped invalid trace file: .ragent/traces/bad.json"],
    )

    text = format_trace_history(result)

    assert "search-20260630T115900Z | search | success" in text
    assert "Warnings:" in text
    assert "- Skipped invalid trace file: .ragent/traces/bad.json" in text


def test_load_trace_view_text_renders_latest_and_recent_history(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_trace(make_trace())
    search_trace = OperationTrace(
        trace_id="search-20260630T000001Z",
        operation="search",
        status="success",
        started_at="2026-06-30T00:00:01Z",
        finished_at="2026-06-30T00:00:02Z",
        steps=[TraceStep(name="read_chunks", description="Read chunks.")],
        metadata={"query": "agent"},
    )
    workspace.write_trace(search_trace)

    text = load_trace_view_text(workspace.root_path)

    assert "Latest trace" in text
    assert "Operation: search" in text
    assert "Recent traces" in text
    assert "search-20260630T000001Z | search | success" in text
    assert "ingest-20260630T000000Z | ingest | success" in text


def test_load_trace_view_text_handles_missing_latest_and_missing_history(
    tmp_path: Path,
) -> None:
    text = load_trace_view_text(tmp_path / ".ragent")

    assert "Latest trace" in text
    assert "No trace found. Run `ragent ingest <path>` first." in text
    assert "Recent traces" in text
    assert "No traces found. Run `ragent ingest <path>` first." in text


def test_load_trace_view_text_shows_history_when_latest_trace_is_missing(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_trace(make_trace())
    workspace.latest_trace_path.unlink()

    text = load_trace_view_text(workspace.root_path)

    assert "Latest trace" in text
    assert "No trace found. Run `ragent ingest <path>` first." in text
    assert "Recent traces" in text
    assert "ingest-20260630T000000Z | ingest | success" in text


def test_load_trace_view_text_shows_history_when_latest_trace_is_corrupt(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_trace(make_trace())
    workspace.latest_trace_path.write_text("{not-json", encoding="utf-8")

    text = load_trace_view_text(workspace.root_path)

    assert "Latest trace" in text
    assert "Trace status error:" in text
    assert "Invalid JSON in latest trace" in text
    assert "Recent traces" in text
    assert "ingest-20260630T000000Z | ingest | success" in text


def test_load_trace_view_text_shows_corrupt_history_warnings(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_trace(make_trace())
    (workspace.traces_dir / "bad.json").write_text("{not-json", encoding="utf-8")

    text = load_trace_view_text(workspace.root_path)

    assert "Recent traces" in text
    assert "ingest-20260630T000000Z | ingest | success" in text
    assert "Warnings:" in text
    assert "Skipped invalid trace file:" in text
    assert "bad.json" in text


def test_load_trace_view_text_limits_recent_history_to_five_traces(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    for index in range(6):
        workspace.write_trace(
            OperationTrace(
                trace_id=f"search-20260630T00000{index}Z",
                operation="search",
                status="success",
                started_at=f"2026-06-30T00:00:0{index}Z",
                finished_at=f"2026-06-30T00:00:0{index}Z",
            )
        )

    text = load_trace_view_text(workspace.root_path)

    assert "search-20260630T000005Z | search | success" in text
    assert "search-20260630T000001Z | search | success" in text
    assert "search-20260630T000000Z | search | success" not in text
