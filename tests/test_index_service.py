import json
from pathlib import Path

import pytest

from ragent_forge.app.models import (
    Document,
    DocumentChunk,
    EmbeddingResult,
    IngestResult,
)
from ragent_forge.app.services.index_service import IndexBuildService
from ragent_forge.app.services.vector_index_service import VectorIndexService
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.core.retrieval.representations import hash_embedding_text


class FakeEmbeddingService:
    provider_name = "openai_embeddings"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(texts)
        return EmbeddingResult(
            provider_name="openai_embeddings",
            model="text-embedding-3-small",
            embeddings=[[float(index + 1), 0.0] for index, _text in enumerate(texts)],
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


def test_index_build_service_uses_structured_representation_and_records_provenance(
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
        embedding_representation="structured_document_text_v1",
    )

    manifest = json.loads(
        workspace.vector_index_manifest_path.read_text(encoding="utf-8")
    )
    records = [
        json.loads(line)
        for line in workspace.vector_index_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.embedding_representation == "structured_document_text_v1"
    assert result.index_input_sha256 == manifest["index_input_sha256"]
    assert manifest["embedding_representation"] == "structured_document_text_v1"
    assert all(
        record["embedding_representation"] == "structured_document_text_v1"
        for record in records
    )
    embedded_texts = [text for batch in embedding_service.calls for text in batch]
    assert [record["text_hash"] for record in records] == [
        hash_embedding_text(text) for text in embedded_texts
    ]
    assert all(
        "Document title:" in text for batch in embedding_service.calls for text in batch
    )


def test_index_build_service_propagates_pdf_sections_across_batches(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(
        [
            DocumentChunk(
                id="/knowledge/book.pdf::chunk-0000",
                document_id="/knowledge/book.pdf",
                text="3\nRandom Vectors in High Dimensions\nOpening text.",
                index=0,
                metadata={
                    "source_path": "/knowledge/book.pdf",
                    "media_type": "application/pdf",
                    "page_start": 53,
                    "page_end": 53,
                    "block_types": ["paragraph"],
                },
            ),
            DocumentChunk(
                id="/knowledge/book.pdf::chunk-0001",
                document_id="/knowledge/book.pdf",
                text="54\nContinuation text.",
                index=1,
                metadata={
                    "source_path": "/knowledge/book.pdf",
                    "media_type": "application/pdf",
                    "page_start": 54,
                    "page_end": 54,
                    "block_types": ["paragraph"],
                },
            ),
        ]
    )
    embedding_service = FakeEmbeddingService()

    result = IndexBuildService(
        workspace,
        embedding_service=embedding_service,
    ).build(
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        batch_size=1,
        embedding_representation="cleaned_pdf_section_text_v1",
    )

    assert result.embedding_representation == "cleaned_pdf_section_text_v1"
    assert len(embedding_service.calls) == 2
    assert all(
        "Section: Random Vectors in High Dimensions" in call[0]
        for call in embedding_service.calls
    )


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
        record.snapshot_id for record in VectorIndexService(workspace).read_index()
    }
    assert index_snapshot_ids == {result.snapshot_id}


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
