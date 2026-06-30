from pathlib import Path

from ragent_forge.app.models import Document, IngestResult
from ragent_forge.app.services.chunk_service import ChunkService, make_preview
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def make_workspace_with_chunks(tmp_path: Path) -> LocalWorkspace:
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
    return workspace


def test_chunk_service_list_chunks_returns_chunks_from_workspace(
    tmp_path: Path,
) -> None:
    workspace = make_workspace_with_chunks(tmp_path)

    chunks = ChunkService(workspace).list_chunks()

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "/knowledge/rag.md::chunk-0000"
    assert chunks[1]["chunk_id"] == "/knowledge/rag.md::chunk-0001"


def test_chunk_service_list_chunks_respects_limit(tmp_path: Path) -> None:
    workspace = make_workspace_with_chunks(tmp_path)

    chunks = ChunkService(workspace).list_chunks(limit=1)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "/knowledge/rag.md::chunk-0000"


def test_chunk_service_get_chunk_returns_existing_chunk(tmp_path: Path) -> None:
    workspace = make_workspace_with_chunks(tmp_path)

    chunk = ChunkService(workspace).get_chunk("/knowledge/rag.md::chunk-0001")

    assert chunk is not None
    assert chunk["text"] == "fghij"


def test_chunk_service_get_chunk_returns_none_for_missing_chunk(
    tmp_path: Path,
) -> None:
    workspace = make_workspace_with_chunks(tmp_path)

    chunk = ChunkService(workspace).get_chunk("missing")

    assert chunk is None


def test_make_preview_removes_newlines_collapses_spaces_and_truncates() -> None:
    preview = make_preview("alpha\n\n beta\tgamma delta", max_length=16)

    assert preview == "alpha beta ga..."
