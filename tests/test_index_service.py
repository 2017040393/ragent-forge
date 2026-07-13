from pathlib import Path

import pytest

from ragent_forge.app.models import Document, EmbeddingResult, IngestResult
from ragent_forge.app.services.index_service import IndexBuildService
from ragent_forge.app.services.vector_index_service import VectorIndexService
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


class FakeEmbeddingService:
    provider_name = "openai_embeddings"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(texts)
        return EmbeddingResult(
            provider_name="openai_embeddings",
            model="text-embedding-3-small",
            embeddings=[
                [float(index + 1), 0.0]
                for index, _text in enumerate(texts)
            ],
            usage={"total_tokens": len(texts)},
            metadata={"base_url": "https://api.openai.com/v1"},
        )


def make_index_build_workspace(tmp_path: Path) -> LocalWorkspace:
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


def test_index_build_service_embeds_chunks_in_batches_and_writes_index(
    tmp_path: Path,
) -> None:
    workspace = make_index_build_workspace(tmp_path)
    embedding_service = FakeEmbeddingService()

    result = IndexBuildService(
        workspace,
        embedding_service=embedding_service,
    ).build(
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        batch_size=2,
    )

    records = workspace.vector_index_path.read_text(encoding="utf-8")
    manifest = workspace.vector_index_manifest_path.read_text(encoding="utf-8")
    assert result.chunk_count == 3
    assert result.embedding_dim == 2
    assert result.embedding_provider == "openai_embeddings"
    assert result.embedding_model == "text-embedding-3-small"
    assert result.batch_size == 2
    assert result.index_path == workspace.vector_index_path
    assert result.manifest_path == workspace.vector_index_manifest_path
    assert embedding_service.calls == [
        ["agent memory agent", "\nretrieval basics\n"],
        ["agent planning"],
    ]
    assert "agent memory agent" not in records
    assert "api_key" not in records
    assert "api_key" not in manifest


def test_index_build_publishes_new_immutable_generation(tmp_path: Path) -> None:
    document = Document(
        id="/knowledge/rag.md",
        text="agent memory agent\nretrieval basics\nagent planning",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunks = SimpleChunker(chunk_size=18, chunk_overlap=0).chunk(document)
    ingest_result = IngestResult(
        source_path="/knowledge",
        documents=[document],
        chunks=chunks,
        skipped_files=[],
        metadata={"chunk_size": 18, "chunk_overlap": 0},
    )
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.commit_ingest_generation(ingest_result, "snapshot-ingest")

    result = IndexBuildService(
        workspace,
        embedding_service=FakeEmbeddingService(),
    ).build(
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        batch_size=2,
    )

    manifest = workspace.read_snapshot_manifest()
    assert manifest is not None
    assert result.snapshot_id == workspace.current_snapshot_id()
    assert result.snapshot_id != "snapshot-ingest"
    assert manifest.parent_snapshot_id == "snapshot-ingest"
    assert manifest.artifacts == [
        "chunks",
        "ingest_summary",
        "vector_index",
        "vector_index_manifest",
    ]
    assert not (
        workspace.generations_dir / "snapshot-ingest" / "vector_index.jsonl"
    ).exists()
    index_snapshot_ids = {
        record.snapshot_id
        for record in VectorIndexService(workspace).read_index()
    }
    assert index_snapshot_ids == {
        result.snapshot_id
    }


def test_index_build_service_rejects_unconfigured_embedding_provider(
    tmp_path: Path,
) -> None:
    workspace = make_index_build_workspace(tmp_path)

    with pytest.raises(RuntimeError, match="embedding provider is not configured"):
        IndexBuildService(workspace, embedding_service=FakeEmbeddingService()).build(
            embedding_provider="none",
            embedding_model=None,
            batch_size=64,
        )


def test_index_build_service_rejects_non_positive_batch_size(
    tmp_path: Path,
) -> None:
    workspace = make_index_build_workspace(tmp_path)

    with pytest.raises(ValueError, match="embedding.batch_size must be greater than 0"):
        IndexBuildService(workspace, embedding_service=FakeEmbeddingService()).build(
            embedding_provider="openai_embeddings",
            embedding_model="text-embedding-3-small",
            batch_size=0,
        )
