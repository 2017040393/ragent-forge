from datetime import UTC, datetime
from pathlib import Path

from ragent_forge.app.models import Document, IngestResult
from ragent_forge.app.services.trace_service import (
    build_ask_retrieval_trace,
    build_ingest_trace,
    build_search_trace,
)
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def test_build_ingest_trace_creates_success_trace_with_metadata() -> None:
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
        skipped_files=["/knowledge/skip.pdf"],
        metadata={"chunk_size": 5, "chunk_overlap": 0},
    )
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 2, tzinfo=UTC)

    trace = build_ingest_trace(
        result=result,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        summary_path=Path(".ragent/ingest/latest_summary.json"),
        started_at=started_at,
        finished_at=finished_at,
    )

    assert trace.trace_id == "ingest-20260630T000000Z"
    assert trace.operation == "ingest"
    assert trace.status == "success"
    assert trace.started_at == "2026-06-30T00:00:00Z"
    assert trace.finished_at == "2026-06-30T00:00:02Z"
    assert [step.name for step in trace.steps] == [
        "load_documents",
        "chunk_documents",
        "write_chunks",
        "write_ingest_summary",
    ]
    assert trace.metadata == {
        "source_path": "/knowledge",
        "document_count": 1,
        "chunk_count": 2,
        "skipped_count": 1,
        "chunk_size": 5,
        "chunk_overlap": 0,
        "chunks_path": str(Path(".ragent/chunks/chunks.jsonl")),
        "summary_path": str(Path(".ragent/ingest/latest_summary.json")),
    }


def test_build_search_trace_creates_success_trace_with_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_search_trace(
        query="agent memory",
        limit=5,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        result_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        started_at=started_at,
        finished_at=finished_at,
    )

    assert trace.trace_id == "search-20260630T000000Z"
    assert trace.operation == "search"
    assert trace.status == "success"
    assert trace.started_at == "2026-06-30T00:00:00Z"
    assert trace.finished_at == "2026-06-30T00:00:01Z"
    assert [step.name for step in trace.steps] == [
        "read_chunks",
        "tokenize_query",
        "score_chunks",
        "rank_results",
        "render_results",
    ]
    assert trace.metadata == {
        "query": "agent memory",
        "limit": 5,
        "scoring_method": "lexical_token_overlap",
        "chunks_path": str(Path(".ragent/chunks/chunks.jsonl")),
        "total_chunks": 7,
        "result_count": 1,
        "result_chunk_ids": ["/knowledge/rag.md::chunk-0002"],
    }


def test_build_ask_retrieval_trace_creates_success_trace_with_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_ask_retrieval_trace(
        question="what is agent memory?",
        limit=3,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        retrieved_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        started_at=started_at,
        finished_at=finished_at,
    )

    assert trace.trace_id == "ask-retrieval-20260630T000000Z"
    assert trace.operation == "ask_retrieval"
    assert trace.status == "success"
    assert trace.started_at == "2026-06-30T00:00:00Z"
    assert trace.finished_at == "2026-06-30T00:00:01Z"
    assert [step.name for step in trace.steps] == [
        "read_chunks",
        "retrieve_context",
        "assemble_context_preview",
        "skip_generation",
        "render_retrieval_preview",
    ]
    assert trace.metadata == {
        "question": "what is agent memory?",
        "limit": 3,
        "retrieval_method": "lexical_token_overlap",
        "chunks_path": str(Path(".ragent/chunks/chunks.jsonl")),
        "total_chunks": 7,
        "retrieved_count": 1,
        "retrieved_chunk_ids": ["/knowledge/rag.md::chunk-0002"],
        "generation_status": "not_implemented",
    }
