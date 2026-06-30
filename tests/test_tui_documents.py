from pathlib import Path

from ragent_forge.app.models import Document, IngestResult, WorkspaceStatus
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.tui.screens.documents import (
    format_workspace_status,
    load_workspace_status_text,
)


def test_format_workspace_status_not_initialized() -> None:
    status = WorkspaceStatus(
        root_path=".ragent",
        exists=False,
        has_chunks=False,
        has_summary=False,
        status="not_initialized",
        chunks_path=".ragent/chunks/chunks.jsonl",
        latest_summary_path=".ragent/ingest/latest_summary.json",
    )

    text = format_workspace_status(status)

    assert "Workspace: .ragent" in text
    assert "Status: not initialized" in text
    assert "Run `ragent ingest <path>`" in text


def test_format_workspace_status_incomplete() -> None:
    status = WorkspaceStatus(
        root_path=".ragent",
        exists=True,
        has_chunks=False,
        has_summary=False,
        status="incomplete",
        chunks_path=".ragent/chunks/chunks.jsonl",
        latest_summary_path=".ragent/ingest/latest_summary.json",
        missing_files=[
            ".ragent/chunks/chunks.jsonl",
            ".ragent/ingest/latest_summary.json",
        ],
    )

    text = format_workspace_status(status)

    assert "Status: incomplete" in text
    assert "Missing files:" in text
    assert "- .ragent/chunks/chunks.jsonl" in text
    assert "- .ragent/ingest/latest_summary.json" in text


def test_format_workspace_status_ready_preserves_zero_chunk_count() -> None:
    status = WorkspaceStatus(
        root_path=".ragent",
        exists=True,
        has_chunks=True,
        has_summary=True,
        status="ready",
        chunks_path=".ragent/chunks/chunks.jsonl",
        latest_summary_path=".ragent/ingest/latest_summary.json",
        summary={
            "source_path": "/knowledge",
            "document_count": 1,
            "chunk_count": 99,
            "skipped_count": 0,
        },
        chunk_count_from_file=0,
    )

    text = format_workspace_status(status)

    assert "Status: ready" in text
    assert "Last ingest source: /knowledge" in text
    assert "Documents: 1" in text
    assert "Chunks: 0" in text
    assert "Skipped files: 0" in text


def test_load_workspace_status_text_handles_corrupt_workspace(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    (workspace_dir / "chunks").mkdir(parents=True)
    (workspace_dir / "ingest").mkdir(parents=True)
    (workspace_dir / "chunks" / "chunks.jsonl").write_text(
        "not-json\n",
        encoding="utf-8",
    )
    (workspace_dir / "ingest" / "latest_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )

    text = load_workspace_status_text(workspace_dir)

    assert "Workspace status error:" in text
    assert "Invalid JSON in chunks file" in text


def test_load_workspace_status_text_shows_recent_chunks_when_ready(
    tmp_path: Path,
) -> None:
    document = Document(
        id="/knowledge/rag.md",
        text="abcdefghij",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunks = SimpleChunker(chunk_size=5, chunk_overlap=0).chunk(document)
    result = IngestResult(
        source_path="/knowledge",
        documents=[document],
        chunks=chunks,
        skipped_files=[],
        metadata={"chunk_size": 5, "chunk_overlap": 0},
    )
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(result.chunks)
    workspace.write_ingest_summary(result)

    text = load_workspace_status_text(workspace.root_path)

    assert "Recent chunks:" in text
    assert "- /knowledge/rag.md::chunk-0000 | /knowledge/rag.md | 0-5 | abcde" in text
