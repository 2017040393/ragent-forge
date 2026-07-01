from datetime import UTC, datetime
from pathlib import Path

from ragent_forge.app.models import Document, GenerationResult, IngestResult
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
        generation_result=GenerationResult(
            provider_name="null",
            status="not_configured",
            answer=None,
        ),
        config_generation_provider="null",
        context_chunk_count=1,
        total_context_chars=128,
        prompt_preview_shown=True,
        max_context_chars=4000,
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
        "generate_answer",
        "render_retrieval_preview",
    ]
    generation_step = trace.steps[3]
    assert generation_step.inputs == {"provider": "null"}
    assert generation_step.outputs == {
        "generation_status": "not_implemented",
        "generation_result_status": "not_configured",
        "answer_generated": False,
    }
    assert trace.metadata == {
        "question": "what is agent memory?",
        "limit": 3,
        "retrieval_method": "lexical_token_overlap",
        "chunks_path": str(Path(".ragent/chunks/chunks.jsonl")),
        "total_chunks": 7,
        "retrieved_count": 1,
        "retrieved_chunk_ids": ["/knowledge/rag.md::chunk-0002"],
        "generation_status": "not_implemented",
        "generation_provider": "null",
        "generation_result_status": "not_configured",
        "answer_generated": False,
        "config_generation_provider": "null",
        "context_chunk_count": 1,
        "total_context_chars": 128,
        "prompt_preview_shown": True,
        "max_context_chars": 4000,
    }


def test_build_ask_retrieval_trace_records_real_generation_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_ask_retrieval_trace(
        question="what is agent memory?",
        limit=3,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        retrieved_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        generation_result=GenerationResult(
            provider_name="openai_responses",
            status="success",
            answer="Generated answer",
            metadata={
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "endpoint": "/responses",
            },
        ),
        config_generation_provider="openai_responses",
        context_chunk_count=1,
        total_context_chars=128,
        prompt_preview_shown=False,
        max_context_chars=4000,
        started_at=started_at,
        finished_at=finished_at,
    )

    generation_step = trace.steps[3]
    assert generation_step.outputs == {
        "generation_status": "generated",
        "generation_result_status": "success",
        "answer_generated": True,
    }
    assert trace.metadata["generation_status"] == "generated"
    assert trace.metadata["generation_provider"] == "openai_responses"
    assert trace.metadata["generation_result_status"] == "success"
    assert trace.metadata["answer_generated"] is True
    assert trace.metadata["model"] == "gpt-4o-mini"
    assert trace.metadata["base_url"] == "https://api.openai.com/v1"
    assert trace.metadata["endpoint"] == "/responses"
    assert trace.metadata["source_count"] == 1
