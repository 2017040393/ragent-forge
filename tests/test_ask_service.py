from pathlib import Path

from ragent_forge.app.models import Document, IngestResult
from ragent_forge.app.services.ask_service import AskService
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def make_ask_workspace(tmp_path: Path) -> LocalWorkspace:
    document = Document(
        id="/knowledge/rag.md",
        text="agent memory agent\nretrieval basics\nagent planning",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunks = SimpleChunker(chunk_size=18, chunk_overlap=0).chunk(document)
    result = IngestResult(
        source_path="/knowledge",
        documents=[document],
        chunks=chunks,
        skipped_files=[],
        metadata={"chunk_size": 18, "chunk_overlap": 0},
    )
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(result.chunks)
    return workspace


def test_ask_service_retrieves_context_without_generation(tmp_path: Path) -> None:
    workspace = make_ask_workspace(tmp_path)

    result = AskService(workspace).retrieve_context("agent memory", limit=1)

    assert result.question == "agent memory"
    assert result.generation_status == "not_implemented"
    assert len(result.results) == 1
    assert result.results[0].chunk_id == "/knowledge/rag.md::chunk-0000"


def test_ask_service_returns_empty_context_when_no_matches(tmp_path: Path) -> None:
    workspace = make_ask_workspace(tmp_path)

    result = AskService(workspace).retrieve_context("no-match", limit=5)

    assert result.question == "no-match"
    assert result.generation_status == "not_implemented"
    assert result.results == []
