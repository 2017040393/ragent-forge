import json
from pathlib import Path

import pytest

from ragent_forge.app.models import Document, IngestResult
from ragent_forge.app.services.vector_index_service import (
    VectorIndexRecord,
    VectorIndexService,
    hash_text,
)
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def make_index_workspace(tmp_path: Path) -> LocalWorkspace:
    document = Document(
        id="/knowledge/rag.md",
        text="agent memory agent\nretrieval basics",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunks = SimpleChunker(chunk_size=20, chunk_overlap=0).chunk(document)
    result = IngestResult(
        source_path="/knowledge",
        documents=[document],
        chunks=chunks,
        skipped_files=[],
        metadata={"chunk_size": 20, "chunk_overlap": 0},
    )
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(result.chunks)
    return workspace


def test_hash_text_uses_deterministic_sha256() -> None:
    assert hash_text("hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e"
        "1b161e5c1fa7425e73043362938b9824"
    )


def test_write_vector_index_jsonl_and_manifest(tmp_path: Path) -> None:
    workspace = make_index_workspace(tmp_path)
    chunks = workspace.read_chunks()
    records = [
        VectorIndexRecord.from_chunk(
            chunk=chunks[0],
            embedding_provider="openai_embeddings",
            embedding_model="text-embedding-3-small",
            embedding=[0.1, 0.2],
        )
    ]
    service = VectorIndexService(workspace)

    result = service.write_index(
        records,
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        chunks_path=workspace.chunks_path,
    )

    lines = workspace.vector_index_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    manifest = json.loads(workspace.vector_index_manifest_path.read_text("utf-8"))
    assert result.index_path == workspace.vector_index_path
    assert record["chunk_id"] == chunks[0]["chunk_id"]
    assert record["embedding_provider"] == "openai_embeddings"
    assert record["embedding_model"] == "text-embedding-3-small"
    assert record["embedding_dim"] == 2
    assert record["embedding"] == [0.1, 0.2]
    assert record["text_hash"] == hash_text(str(chunks[0]["text"]))
    assert "text" not in record
    assert "api_key" not in record
    assert manifest["chunk_count"] == 1
    assert manifest["embedding_dim"] == 2
    assert manifest["embedding_representation"] == "raw_chunk_text_v1"
    assert len(manifest["index_input_sha256"]) == 64


def test_read_vector_index_jsonl(tmp_path: Path) -> None:
    workspace = make_index_workspace(tmp_path)
    chunk = workspace.read_chunks()[0]
    record = VectorIndexRecord.from_chunk(
        chunk=chunk,
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        embedding=[0.1, 0.2],
    )
    service = VectorIndexService(workspace)
    service.write_index(
        [record],
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        chunks_path=workspace.chunks_path,
    )

    records = service.read_index()

    assert records == [record]


def test_read_vector_index_rejects_workspace_snapshot_mismatch(
    tmp_path: Path,
) -> None:
    workspace = make_index_workspace(tmp_path)
    chunk = workspace.read_chunks()[0]
    workspace.commit_snapshot("snapshot-current", "/knowledge", 1)
    service = VectorIndexService(workspace)
    service.write_index(
        [
            VectorIndexRecord.from_chunk(
                chunk={**chunk, "snapshot_id": "snapshot-old"},
                embedding_provider="openai_embeddings",
                embedding_model="text-embedding-3-small",
                embedding=[0.1, 0.2],
            )
        ],
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        chunks_path=workspace.chunks_path,
        snapshot_id="snapshot-current",
    )

    with pytest.raises(ValueError, match="Vector index snapshot mismatch"):
        service.read_index()


def test_read_empty_vector_index_accepts_matching_snapshot_manifest(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.commit_snapshot("snapshot-current", "/knowledge", 0)
    service = VectorIndexService(workspace)
    service.write_index(
        [],
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        chunks_path=workspace.chunks_path,
        snapshot_id="snapshot-current",
    )

    assert service.read_index() == []


def test_read_vector_index_also_validates_manifest_snapshot(
    tmp_path: Path,
) -> None:
    workspace = make_index_workspace(tmp_path)
    chunk = workspace.read_chunks()[0]
    workspace.commit_snapshot("snapshot-current", "/knowledge", 1)
    record = VectorIndexRecord.from_chunk(
        chunk={**chunk, "snapshot_id": "snapshot-current"},
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        embedding=[0.1, 0.2],
    )
    service = VectorIndexService(workspace)
    service.write_index(
        [record],
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        chunks_path=workspace.chunks_path,
        snapshot_id="snapshot-old",
    )

    with pytest.raises(
        ValueError,
        match="Vector index manifest snapshot mismatch",
    ):
        service.read_index()


def test_read_vector_index_raises_clear_error_for_corrupt_jsonl(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    workspace.vector_index_path.write_text("{ok}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in vector index"):
        VectorIndexService(workspace).read_index()
