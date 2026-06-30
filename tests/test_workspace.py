import json
from pathlib import Path

import pytest

from ragent_forge.app.models import Document, IngestResult, OperationTrace, TraceStep
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def make_ingest_result() -> IngestResult:
    document = Document(
        id="/knowledge/rag.md",
        text="abcdefghij",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunks = SimpleChunker(chunk_size=5, chunk_overlap=0).chunk(document)
    return IngestResult(
        source_path="/knowledge",
        documents=[document],
        chunks=chunks,
        skipped_files=["/knowledge/skip.pdf"],
        metadata={"chunk_size": 5, "chunk_overlap": 0},
    )


def test_workspace_exists_before_and_after_ensure_exists(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")

    assert workspace.exists() is False

    workspace.ensure_exists()

    assert workspace.exists() is True


def test_workspace_ensure_exists_creates_required_directories(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")

    workspace.ensure_exists()

    assert workspace.root_path.is_dir()
    assert workspace.chunks_dir.is_dir()
    assert workspace.ingest_dir.is_dir()
    assert workspace.traces_dir.is_dir()


def test_write_chunks_writes_valid_jsonl(tmp_path: Path) -> None:
    document = Document(
        id="/knowledge/rag.md",
        text="abcdefghij",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunks = SimpleChunker(chunk_size=5, chunk_overlap=0).chunk(document)
    workspace = LocalWorkspace(tmp_path / ".ragent")

    workspace.write_chunks(chunks)

    lines = workspace.chunks_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    assert records[0]["chunk_id"] == "/knowledge/rag.md::chunk-0000"
    assert records[0]["document_id"] == "/knowledge/rag.md"
    assert records[0]["text"] == "abcde"
    assert records[0]["source_path"] == "/knowledge/rag.md"
    assert records[0]["start_char"] == 0
    assert records[0]["end_char"] == 5
    assert records[0]["metadata"]["source_path"] == "/knowledge/rag.md"


def test_write_ingest_summary_writes_valid_json(tmp_path: Path) -> None:
    result = make_ingest_result()
    workspace = LocalWorkspace(tmp_path / ".ragent")

    workspace.write_ingest_summary(result)

    summary = json.loads(workspace.latest_summary_path.read_text(encoding="utf-8"))
    assert summary == {
        "source_path": "/knowledge",
        "document_count": 1,
        "chunk_count": 2,
        "skipped_count": 1,
        "skipped_files": ["/knowledge/skip.pdf"],
        "metadata": {"chunk_size": 5, "chunk_overlap": 0},
    }


def test_read_chunks_reads_valid_jsonl(tmp_path: Path) -> None:
    result = make_ingest_result()
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(result.chunks)

    records = workspace.read_chunks()

    assert len(records) == 2
    assert records[0]["chunk_id"] == "/knowledge/rag.md::chunk-0000"


def test_read_ingest_summary_reads_valid_json(tmp_path: Path) -> None:
    result = make_ingest_result()
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_ingest_summary(result)

    summary = workspace.read_ingest_summary()

    assert summary["source_path"] == "/knowledge"
    assert summary["document_count"] == 1


def test_workspace_status_not_initialized_when_missing(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")

    status = workspace.status()

    assert status.status == "not_initialized"
    assert status.exists is False
    assert status.has_chunks is False
    assert status.has_summary is False
    assert status.missing_files == []


def test_workspace_status_ready_with_zero_chunks_reports_zero_count(
    tmp_path: Path,
) -> None:
    result = IngestResult(
        source_path="/knowledge",
        documents=[],
        chunks=[],
        skipped_files=[],
        metadata={"chunk_size": 5, "chunk_overlap": 0},
    )
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(result.chunks)
    workspace.write_ingest_summary(result)

    status = workspace.status()

    assert status.status == "ready"
    assert status.chunk_count_from_file == 0


def test_workspace_status_incomplete_when_files_are_missing(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()

    status = workspace.status()

    assert status.status == "incomplete"
    assert status.exists is True
    assert status.has_chunks is False
    assert status.has_summary is False
    assert status.missing_files == [
        str(workspace.chunks_path),
        str(workspace.latest_summary_path),
    ]


def test_workspace_status_ready_after_writing_chunks_and_summary(
    tmp_path: Path,
) -> None:
    result = make_ingest_result()
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(result.chunks)
    workspace.write_ingest_summary(result)

    status = workspace.status()

    assert status.status == "ready"
    assert status.exists is True
    assert status.has_chunks is True
    assert status.has_summary is True
    assert status.summary["document_count"] == 1
    assert status.chunk_count_from_file == 2
    assert status.missing_files == []


def test_read_chunks_raises_clear_error_for_invalid_jsonl(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    workspace.chunks_path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in chunks file"):
        workspace.read_chunks()


def test_read_ingest_summary_raises_clear_error_for_invalid_json(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    workspace.latest_summary_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in ingest summary"):
        workspace.read_ingest_summary()


def test_write_trace_writes_trace_and_latest_trace_json(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    trace = OperationTrace(
        trace_id="ingest-20260630T000000Z",
        operation="ingest",
        status="success",
        started_at="2026-06-30T00:00:00Z",
        finished_at="2026-06-30T00:00:01Z",
        steps=[
            TraceStep(
                name="load_documents",
                description="Load supported source documents.",
            )
        ],
        metadata={"document_count": 1},
    )

    latest_trace_path = workspace.write_trace(trace)

    trace_path = workspace.traces_dir / "ingest-20260630T000000Z.json"
    assert trace_path.is_file()
    assert latest_trace_path == workspace.latest_trace_path
    assert workspace.latest_trace_path.is_file()

    trace_record = json.loads(trace_path.read_text(encoding="utf-8"))
    latest_record = json.loads(workspace.latest_trace_path.read_text(encoding="utf-8"))
    assert trace_record["trace_id"] == "ingest-20260630T000000Z"
    assert latest_record == trace_record


def test_read_latest_trace_reads_valid_trace_json(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    trace = OperationTrace(
        trace_id="ingest-20260630T000000Z",
        operation="ingest",
        status="success",
        started_at="2026-06-30T00:00:00Z",
        finished_at="2026-06-30T00:00:01Z",
        metadata={"chunk_count": 2},
    )
    workspace.write_trace(trace)

    record = workspace.read_latest_trace()

    assert record["trace_id"] == "ingest-20260630T000000Z"
    assert record["metadata"]["chunk_count"] == 2


def test_has_latest_trace_reports_missing_and_existing_trace(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")

    assert workspace.has_latest_trace() is False

    workspace.write_trace(
        OperationTrace(
            trace_id="ingest-20260630T000000Z",
            operation="ingest",
            status="success",
            started_at="2026-06-30T00:00:00Z",
        )
    )

    assert workspace.has_latest_trace() is True


def test_read_latest_trace_raises_clear_error_for_invalid_json(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    workspace.latest_trace_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in latest trace"):
        workspace.read_latest_trace()
