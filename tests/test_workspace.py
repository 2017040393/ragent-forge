import json
from pathlib import Path

from ragent_forge.app.models import Document, IngestResult
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def test_workspace_ensure_exists_creates_required_directories(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")

    workspace.ensure_exists()

    assert workspace.root_path.is_dir()
    assert workspace.chunks_dir.is_dir()
    assert workspace.ingest_dir.is_dir()


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
    result = IngestResult(
        source_path="/knowledge",
        documents=[
            Document(
                id="/knowledge/rag.md",
                text="abcdefghij",
                metadata={"source_path": "/knowledge/rag.md"},
            )
        ],
        chunks=[],
        skipped_files=["/knowledge/skip.pdf"],
        metadata={"chunk_size": 5, "chunk_overlap": 0},
    )
    workspace = LocalWorkspace(tmp_path / ".ragent")

    workspace.write_ingest_summary(result)

    summary = json.loads(workspace.latest_summary_path.read_text(encoding="utf-8"))
    assert summary == {
        "source_path": "/knowledge",
        "document_count": 1,
        "chunk_count": 0,
        "skipped_count": 1,
        "skipped_files": ["/knowledge/skip.pdf"],
        "metadata": {"chunk_size": 5, "chunk_overlap": 0},
    }
