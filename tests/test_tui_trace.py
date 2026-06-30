from pathlib import Path

from ragent_forge.app.models import OperationTrace, TraceStep
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.tui.screens.trace import format_latest_trace, load_latest_trace_text


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
