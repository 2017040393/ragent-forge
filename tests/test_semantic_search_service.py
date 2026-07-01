from pathlib import Path

import pytest

from ragent_forge.app.models import Document, EmbeddingResult, IngestResult
from ragent_forge.app.services.semantic_search_service import (
    SemanticSearchService,
    cosine_similarity,
)
from ragent_forge.app.services.vector_index_service import (
    VectorIndexRecord,
    VectorIndexService,
)
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


class FakeEmbeddingService:
    provider_name = "openai_embeddings"

    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(
            provider_name=self.provider_name,
            model="text-embedding-3-small",
            embeddings=[self.embedding for _ in texts],
            metadata={
                "base_url": "https://api.openai.com/v1",
                "endpoint": "/embeddings",
            },
        )


def make_semantic_workspace(tmp_path: Path) -> LocalWorkspace:
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
    chunk_records = workspace.read_chunks()
    index_records = [
        VectorIndexRecord.from_chunk(
            chunk=chunk_records[0],
            embedding_provider="openai_embeddings",
            embedding_model="text-embedding-3-small",
            embedding=[1.0, 0.0],
        ),
        VectorIndexRecord.from_chunk(
            chunk=chunk_records[1],
            embedding_provider="openai_embeddings",
            embedding_model="text-embedding-3-small",
            embedding=[0.0, 1.0],
        ),
        VectorIndexRecord.from_chunk(
            chunk=chunk_records[2],
            embedding_provider="openai_embeddings",
            embedding_model="text-embedding-3-small",
            embedding=[0.5, 0.5],
        ),
    ]
    VectorIndexService(workspace).write_index(
        index_records,
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        chunks_path=workspace.chunks_path,
    )
    return workspace


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_returns_zero_for_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_dimension_mismatch_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Embedding dimensions do not match"):
        cosine_similarity([1.0], [1.0, 0.0])


def test_semantic_search_ranks_by_cosine_similarity(tmp_path: Path) -> None:
    workspace = make_semantic_workspace(tmp_path)
    service = SemanticSearchService(
        workspace,
        embedding_service=FakeEmbeddingService([1.0, 0.0]),
    )

    results = service.search("agent memory", limit=3)

    assert [result.chunk_id for result in results] == [
        "/knowledge/rag.md::chunk-0000",
        "/knowledge/rag.md::chunk-0002",
        "/knowledge/rag.md::chunk-0001",
    ]
    assert results[0].score > results[1].score > results[2].score
    assert results[0].text == "agent memory agent"


def test_semantic_search_respects_limit(tmp_path: Path) -> None:
    workspace = make_semantic_workspace(tmp_path)
    service = SemanticSearchService(
        workspace,
        embedding_service=FakeEmbeddingService([1.0, 0.0]),
    )

    results = service.search("agent memory", limit=1)

    assert len(results) == 1
    assert results[0].chunk_id == "/knowledge/rag.md::chunk-0000"


def test_semantic_search_missing_vector_index_fails_clearly(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    service = SemanticSearchService(
        workspace,
        embedding_service=FakeEmbeddingService([1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="vector index not found"):
        service.search("agent memory")


def test_semantic_search_error_does_not_leak_api_key(tmp_path: Path) -> None:
    workspace = make_semantic_workspace(tmp_path)
    service = SemanticSearchService(
        workspace,
        embedding_service=FakeEmbeddingService([1.0]),
    )

    with pytest.raises(ValueError) as exc_info:
        service.search("agent memory")

    assert "sk-test-secret" not in str(exc_info.value)
