from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ragent_forge.app.models import (
    GenerationResult,
    IngestResult,
    OperationTrace,
    RagTrace,
    TraceStep,
)


class TraceService:
    def create_empty_trace(self, query: str) -> RagTrace:
        return RagTrace(query=query, metadata={"status": "stub"})


def build_ingest_trace(
    result: IngestResult,
    chunks_path: Path,
    summary_path: Path,
    started_at: datetime,
    finished_at: datetime,
) -> OperationTrace:
    started_at_utc = _as_utc(started_at)
    finished_at_utc = _as_utc(finished_at)
    metadata = {
        "source_path": result.source_path,
        "document_count": result.document_count,
        "chunk_count": result.chunk_count,
        "skipped_count": result.skipped_count,
        "chunk_size": result.metadata["chunk_size"],
        "chunk_overlap": result.metadata["chunk_overlap"],
        "chunks_path": str(chunks_path),
        "summary_path": str(summary_path),
    }
    return OperationTrace(
        trace_id=f"ingest-{started_at_utc.strftime('%Y%m%dT%H%M%SZ')}",
        operation="ingest",
        status="success",
        started_at=_format_timestamp(started_at_utc),
        finished_at=_format_timestamp(finished_at_utc),
        steps=[
            TraceStep(
                name="load_documents",
                description="Load supported Markdown/TXT documents.",
                inputs={"source_path": result.source_path},
                outputs={
                    "document_count": result.document_count,
                    "skipped_count": result.skipped_count,
                },
            ),
            TraceStep(
                name="chunk_documents",
                description="Split loaded documents into deterministic chunks.",
                inputs={
                    "document_count": result.document_count,
                    "chunk_size": result.metadata["chunk_size"],
                    "chunk_overlap": result.metadata["chunk_overlap"],
                },
                outputs={"chunk_count": result.chunk_count},
            ),
            TraceStep(
                name="write_chunks",
                description="Persist chunk records to the local workspace.",
                inputs={"chunk_count": result.chunk_count},
                outputs={"chunks_path": str(chunks_path)},
            ),
            TraceStep(
                name="write_ingest_summary",
                description="Persist the latest ingestion summary.",
                inputs={
                    "document_count": result.document_count,
                    "chunk_count": result.chunk_count,
                    "skipped_count": result.skipped_count,
                },
                outputs={"summary_path": str(summary_path)},
            ),
        ],
        metadata=metadata,
    )


def build_search_trace(
    query: str,
    limit: int,
    chunks_path: Path,
    total_chunks: int,
    result_chunk_ids: list[str],
    started_at: datetime,
    finished_at: datetime,
) -> OperationTrace:
    started_at_utc = _as_utc(started_at)
    finished_at_utc = _as_utc(finished_at)
    metadata = {
        "query": query,
        "limit": limit,
        "scoring_method": "lexical_token_overlap",
        "chunks_path": str(chunks_path),
        "total_chunks": total_chunks,
        "result_count": len(result_chunk_ids),
        "result_chunk_ids": result_chunk_ids,
    }
    return OperationTrace(
        trace_id=f"search-{started_at_utc.strftime('%Y%m%dT%H%M%SZ')}",
        operation="search",
        status="success",
        started_at=_format_timestamp(started_at_utc),
        finished_at=_format_timestamp(finished_at_utc),
        steps=[
            TraceStep(
                name="read_chunks",
                description="Read chunk records from the local workspace.",
                inputs={"chunks_path": str(chunks_path)},
                outputs={"total_chunks": total_chunks},
            ),
            TraceStep(
                name="tokenize_query",
                description="Normalize and tokenize the lexical search query.",
                inputs={"query": query},
                outputs={"scoring_method": "lexical_token_overlap"},
            ),
            TraceStep(
                name="score_chunks",
                description="Score chunks by lexical token overlap.",
                inputs={"total_chunks": total_chunks},
                outputs={"matched_chunks": len(result_chunk_ids)},
            ),
            TraceStep(
                name="rank_results",
                description="Sort results by score and deterministic chunk id.",
                inputs={"matched_chunks": len(result_chunk_ids), "limit": limit},
                outputs={"result_chunk_ids": result_chunk_ids},
            ),
            TraceStep(
                name="render_results",
                description="Render search results in the CLI.",
                inputs={"result_count": len(result_chunk_ids)},
                outputs={"status": "success"},
            ),
        ],
        metadata=metadata,
    )


def build_ask_retrieval_trace(
    question: str,
    limit: int,
    chunks_path: Path,
    total_chunks: int,
    retrieved_chunk_ids: list[str],
    generation_result: GenerationResult,
    config_generation_provider: str,
    context_chunk_count: int,
    total_context_chars: int,
    prompt_preview_shown: bool,
    max_context_chars: int,
    started_at: datetime,
    finished_at: datetime,
) -> OperationTrace:
    started_at_utc = _as_utc(started_at)
    finished_at_utc = _as_utc(finished_at)
    metadata = {
        "question": question,
        "limit": limit,
        "retrieval_method": "lexical_token_overlap",
        "chunks_path": str(chunks_path),
        "total_chunks": total_chunks,
        "retrieved_count": len(retrieved_chunk_ids),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "generation_status": "not_implemented",
        "generation_provider": generation_result.provider_name,
        "generation_result_status": generation_result.status,
        "answer_generated": generation_result.answer is not None,
        "config_generation_provider": config_generation_provider,
        "context_chunk_count": context_chunk_count,
        "total_context_chars": total_context_chars,
        "prompt_preview_shown": prompt_preview_shown,
        "max_context_chars": max_context_chars,
    }
    return OperationTrace(
        trace_id=f"ask-retrieval-{started_at_utc.strftime('%Y%m%dT%H%M%SZ')}",
        operation="ask_retrieval",
        status="success",
        started_at=_format_timestamp(started_at_utc),
        finished_at=_format_timestamp(finished_at_utc),
        steps=[
            TraceStep(
                name="read_chunks",
                description="Read chunk records from the local workspace.",
                inputs={"chunks_path": str(chunks_path)},
                outputs={"total_chunks": total_chunks},
            ),
            TraceStep(
                name="retrieve_context",
                description="Retrieve context chunks with lexical search.",
                inputs={"question": question, "limit": limit},
                outputs={"retrieved_count": len(retrieved_chunk_ids)},
            ),
            TraceStep(
                name="assemble_context_preview",
                description="Prepare retrieved chunks for inspectable preview.",
                inputs={"retrieved_chunk_ids": retrieved_chunk_ids},
                outputs={"preview_count": len(retrieved_chunk_ids)},
            ),
            TraceStep(
                name="generate_answer",
                description=(
                    "Run the configured generation provider; the null provider "
                    "returns no answer."
                ),
                inputs={"provider": generation_result.provider_name},
                outputs={
                    "generation_status": "not_implemented",
                    "generation_result_status": generation_result.status,
                    "answer_generated": generation_result.answer is not None,
                },
            ),
            TraceStep(
                name="render_retrieval_preview",
                description="Render the retrieval-only ask preview in the CLI.",
                inputs={"retrieved_count": len(retrieved_chunk_ids)},
                outputs={"status": "success"},
            ),
        ],
        metadata=metadata,
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")
