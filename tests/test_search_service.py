import json
from pathlib import Path

import pytest

from ragent_forge.app.models import Document, IngestResult
from ragent_forge.app.services.search_service import (
    BM25SearchService,
    LexicalSearchService,
    score_text,
    tokenize,
)
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def make_search_workspace(tmp_path: Path) -> LocalWorkspace:
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


def make_bm25_workspace(tmp_path: Path) -> LocalWorkspace:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    records = [
        {
            "chunk_id": "a::chunk-0000",
            "document_id": "a",
            "source_path": "a.md",
            "start_char": 0,
            "end_char": 24,
            "text": "agent agent memory",
            "metadata": {"source_path": "a.md", "section": "memory"},
        },
        {
            "chunk_id": "b::chunk-0000",
            "document_id": "b",
            "source_path": "b.md",
            "start_char": 25,
            "end_char": 52,
            "text": "agent planning",
            "metadata": {"source_path": "b.md", "section": "planning"},
        },
        {
            "chunk_id": "c::chunk-0000",
            "document_id": "c",
            "source_path": "c.md",
            "start_char": 53,
            "end_char": 79,
            "text": "retrieval basics",
            "metadata": {"source_path": "c.md", "section": "retrieval"},
        },
    ]
    workspace.chunks_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return workspace


def test_tokenize_lowercases_and_extracts_word_tokens() -> None:
    assert tokenize("Agent-memory, RAG_101!") == ["agent", "memory", "rag_101"]


def test_score_text_returns_positive_score_for_matching_terms() -> None:
    score = score_text(["agent", "memory"], tokenize("agent agent memory"))

    assert score == 3.0


def test_score_text_returns_zero_when_no_terms_match() -> None:
    score = score_text(["agent"], tokenize("retrieval basics"))

    assert score == 0.0


def test_lexical_search_returns_matching_chunks(tmp_path: Path) -> None:
    workspace = make_search_workspace(tmp_path)

    results = LexicalSearchService(workspace).search("memory")

    assert len(results) == 1
    assert results[0].chunk_id == "/knowledge/rag.md::chunk-0000"
    assert results[0].score > 0


def test_lexical_search_sorts_by_score_descending(tmp_path: Path) -> None:
    workspace = make_search_workspace(tmp_path)

    results = LexicalSearchService(workspace).search("agent")

    assert [result.chunk_id for result in results] == [
        "/knowledge/rag.md::chunk-0000",
        "/knowledge/rag.md::chunk-0002",
    ]
    assert results[0].score > results[1].score


def test_lexical_search_sorts_ties_by_chunk_id(tmp_path: Path) -> None:
    workspace = make_search_workspace(tmp_path)

    results = LexicalSearchService(workspace).search("retrieval planning")

    assert [result.chunk_id for result in results] == [
        "/knowledge/rag.md::chunk-0001",
        "/knowledge/rag.md::chunk-0002",
    ]
    assert results[0].score == results[1].score


def test_lexical_search_respects_limit(tmp_path: Path) -> None:
    workspace = make_search_workspace(tmp_path)

    results = LexicalSearchService(workspace).search("agent", limit=1)

    assert len(results) == 1
    assert results[0].chunk_id == "/knowledge/rag.md::chunk-0000"


def test_lexical_search_counts_chunks(tmp_path: Path) -> None:
    workspace = make_search_workspace(tmp_path)

    assert LexicalSearchService(workspace).count_chunks() == 3


def test_lexical_search_empty_query_returns_empty_list(tmp_path: Path) -> None:
    workspace = make_search_workspace(tmp_path)

    assert LexicalSearchService(workspace).search("   ") == []


def test_bm25_search_returns_matching_chunks_ranked_by_score(tmp_path: Path) -> None:
    workspace = make_bm25_workspace(tmp_path)

    results = BM25SearchService(workspace).search("agent memory")

    assert [result.chunk_id for result in results] == [
        "a::chunk-0000",
        "b::chunk-0000",
    ]
    assert results[0].score > results[1].score


def test_bm25_search_repeated_document_terms_affect_ranking(tmp_path: Path) -> None:
    workspace = make_bm25_workspace(tmp_path)

    results = BM25SearchService(workspace).search("agent")

    assert [result.chunk_id for result in results[:2]] == [
        "a::chunk-0000",
        "b::chunk-0000",
    ]
    assert results[0].score > results[1].score


def test_bm25_search_rare_terms_have_higher_idf_impact(tmp_path: Path) -> None:
    workspace = make_bm25_workspace(tmp_path)

    results = BM25SearchService(workspace).search("agent retrieval")

    retrieval_result = next(
        result for result in results if result.chunk_id == "c::chunk-0000"
    )
    planning_result = next(
        result for result in results if result.chunk_id == "b::chunk-0000"
    )
    assert retrieval_result.score > planning_result.score


def test_bm25_search_empty_query_returns_empty_list(tmp_path: Path) -> None:
    workspace = make_bm25_workspace(tmp_path)

    assert BM25SearchService(workspace).search("   ") == []


def test_bm25_search_rejects_negative_limit(tmp_path: Path) -> None:
    workspace = make_bm25_workspace(tmp_path)

    with pytest.raises(ValueError, match="limit must be greater than or equal to 0"):
        BM25SearchService(workspace).search("agent", limit=-1)


def test_bm25_search_sorts_ties_by_chunk_id(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    records = [
        {
            "chunk_id": "b::chunk-0000",
            "document_id": "b",
            "source_path": "b.md",
            "text": "agent memory",
            "metadata": {"source_path": "b.md"},
        },
        {
            "chunk_id": "a::chunk-0000",
            "document_id": "a",
            "source_path": "a.md",
            "text": "agent memory",
            "metadata": {"source_path": "a.md"},
        },
    ]
    workspace.chunks_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    results = BM25SearchService(workspace).search("agent")

    assert [result.chunk_id for result in results] == [
        "a::chunk-0000",
        "b::chunk-0000",
    ]


def test_bm25_search_preserves_result_fields(tmp_path: Path) -> None:
    workspace = make_bm25_workspace(tmp_path)

    result = BM25SearchService(workspace).search("memory")[0]

    assert result.chunk_id == "a::chunk-0000"
    assert result.document_id == "a"
    assert result.source_path == "a.md"
    assert result.start_char == 0
    assert result.end_char == 24
    assert result.metadata == {"source_path": "a.md", "section": "memory"}
    assert result.text == "agent agent memory"
