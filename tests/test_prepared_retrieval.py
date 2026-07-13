from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import pytest

from ragent_forge.app.models import EmbeddingResult
from ragent_forge.app.ports import ApplicationWorkspace, RetrievalWorkspace
from ragent_forge.app.services.prepared_retrieval import PreparedStateCache
from ragent_forge.app.services.search_service import (
    BM25SearchService,
    LexicalSearchService,
    tokenize,
)
from ragent_forge.app.services.semantic_search_service import SemanticSearchService
from ragent_forge.app.services.vector_index_service import VectorIndexRecord
from ragent_forge.composition import build_retrieval_runtime
from ragent_forge.core.retrieval.contracts import ChunkRecord


class CountingWorkspace:
    def __init__(self, root_path: Path | None = None) -> None:
        self.root_path = root_path or Path(".benchmark-workspace")
        self.records: list[ChunkRecord] = [
            {
                "schema_version": 2,
                "snapshot_id": "snapshot-1",
                "chunk_id": "doc::chunk-0000",
                "document_id": "doc",
                "text": "agent memory retrieval",
                "source_path": "doc.md",
                "start_char": None,
                "end_char": None,
                "metadata": {},
                "source_kind": "document",
                "provenance": None,
                "authority": "source",
                "freshness": None,
                "lifecycle": "regenerable",
            },
            {
                "schema_version": 2,
                "snapshot_id": "snapshot-1",
                "chunk_id": "doc::chunk-0001",
                "document_id": "doc",
                "text": "other context",
                "source_path": "doc.md",
                "start_char": None,
                "end_char": None,
                "metadata": {},
                "source_kind": "document",
                "provenance": None,
                "authority": "source",
                "freshness": None,
                "lifecycle": "regenerable",
            },
        ]
        self.snapshot_id: str | None = "snapshot-1"
        self.read_chunks_calls = 0

    def read_chunks(self) -> list[ChunkRecord]:
        self.read_chunks_calls += 1
        return list(self.records)

    def current_snapshot_id(self) -> str | None:
        return self.snapshot_id


def test_lexical_cache_reuses_tokenization_and_invalidates_on_snapshot_change() -> None:
    workspace = CountingWorkspace()
    tokenize_calls = 0

    def counting_tokenize(text: str) -> list[str]:
        nonlocal tokenize_calls
        tokenize_calls += 1
        return tokenize(text)

    cache = PreparedStateCache(counting_tokenize)
    service = LexicalSearchService(workspace, prepared_state_cache=cache)

    cold_results = service.search("agent memory", limit=2)
    warm_results = service.search("agent memory", limit=2)

    assert [result.chunk_id for result in cold_results] == [
        "doc::chunk-0000"
    ]
    assert [result.chunk_id for result in warm_results] == [
        "doc::chunk-0000"
    ]
    assert workspace.read_chunks_calls == 1
    assert tokenize_calls == len(workspace.records)
    stats = cache.stats()
    assert stats.chunk_loads == 1
    assert stats.warm_hits == 1
    assert stats.invalidations == 0
    assert stats.last_chunk_load_latency_ms is not None

    workspace.snapshot_id = "snapshot-2"
    workspace.records[0]["text"] = "new snapshot facts"
    service.search("new snapshot", limit=2)

    assert workspace.read_chunks_calls == 2
    assert tokenize_calls == len(workspace.records) * 2
    assert cache.stats().invalidations == 1
    assert cache.stats().snapshot_id == "snapshot-2"


def test_bm25_and_lexical_services_can_share_prepared_chunks() -> None:
    workspace = CountingWorkspace()
    cache = PreparedStateCache(tokenize)
    lexical = LexicalSearchService(workspace, prepared_state_cache=cache)
    bm25 = BM25SearchService(workspace, prepared_state_cache=cache)

    lexical.search("agent", limit=2)
    bm25.search("agent", limit=2)

    assert workspace.read_chunks_calls == 1
    assert cache.stats().chunk_loads == 1
    assert cache.stats().warm_hits == 1


def test_composition_reuses_cache_across_runtime_builds_and_snapshots(
    tmp_path: Path,
) -> None:
    workspace = CountingWorkspace(tmp_path / ".ragent")
    application_workspace = cast(ApplicationWorkspace, workspace)

    lexical = build_retrieval_runtime(
        application_workspace,
        "lexical",
        limit=2,
    )
    lexical.retrieval_engine.run("agent", 2)
    bm25 = build_retrieval_runtime(
        application_workspace,
        "bm25",
        limit=2,
    )
    bm25.retrieval_engine.run("agent", 2)

    assert lexical.prepared_state_cache is bm25.prepared_state_cache
    assert workspace.read_chunks_calls == 1
    assert bm25.prepared_state_cache is not None
    assert bm25.prepared_state_cache.stats().warm_hits == 1

    workspace.snapshot_id = "snapshot-2"
    refreshed = build_retrieval_runtime(
        application_workspace,
        "lexical",
        limit=2,
    )
    refreshed.retrieval_engine.run("agent", 2)

    assert refreshed.prepared_state_cache is bm25.prepared_state_cache
    assert workspace.read_chunks_calls == 2
    assert refreshed.prepared_state_cache is not None
    assert refreshed.prepared_state_cache.stats().invalidations == 1


def test_warm_query_avoids_cold_workspace_load_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = CountingWorkspace()
    original_read_chunks = workspace.read_chunks

    def slow_read_chunks() -> list[ChunkRecord]:
        time.sleep(0.02)
        return original_read_chunks()

    monkeypatch.setattr(workspace, "read_chunks", slow_read_chunks)
    service = BM25SearchService(workspace)

    started = time.perf_counter()
    service.search("agent", limit=2)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    service.search("agent", limit=2)
    warm_seconds = time.perf_counter() - started

    assert warm_seconds < cold_seconds
    assert workspace.read_chunks_calls == 1


class FakeEmbeddingService:
    provider_name = "test"

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(
            provider_name=self.provider_name,
            model="test-model",
            embeddings=[[1.0, 0.0] for _ in texts],
        )


def test_semantic_cache_reuses_vector_records_and_invalidates_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = CountingWorkspace()
    cache = PreparedStateCache(lambda _text: [])
    service = SemanticSearchService(
        cast(RetrievalWorkspace, workspace),
        FakeEmbeddingService(),
        prepared_state_cache=cache,
    )
    read_index_calls = 0
    record = VectorIndexRecord(
        snapshot_id="snapshot-1",
        chunk_id="doc::chunk-0000",
        document_id="doc",
        source_path="doc.md",
        embedding_provider="test",
        embedding_model="test-model",
        embedding_dim=2,
        embedding=[1.0, 0.0],
        text_hash="hash",
    )

    def read_index() -> list[VectorIndexRecord]:
        nonlocal read_index_calls
        read_index_calls += 1
        return [record]

    monkeypatch.setattr(service.vector_index_service, "read_index", read_index)

    service.search("agent", limit=1)
    service.search("agent", limit=1)
    assert workspace.read_chunks_calls == 1
    assert read_index_calls == 1
    assert cache.stats().vector_loads == 1

    workspace.snapshot_id = "snapshot-2"
    service.search("agent", limit=1)
    assert workspace.read_chunks_calls == 2
    assert read_index_calls == 2
    assert cache.stats().invalidations == 1
