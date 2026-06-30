from pathlib import Path

from ragent_forge.app.models import Document, IngestResult
from ragent_forge.app.services.search_service import (
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


def test_lexical_search_empty_query_returns_empty_list(tmp_path: Path) -> None:
    workspace = make_search_workspace(tmp_path)

    assert LexicalSearchService(workspace).search("   ") == []
