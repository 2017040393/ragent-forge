from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    retrieval_mode: str = "lexical",
    retrieval_method: str = "lexical_token_overlap",
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    index_path: Path | None = None,
) -> OperationTrace:
    started_at_utc = _as_utc(started_at)
    finished_at_utc = _as_utc(finished_at)
    metadata = {
        "query": query,
        "limit": limit,
        "retrieval_mode": retrieval_mode,
        "scoring_method": retrieval_method,
        "chunks_path": str(chunks_path),
        "total_chunks": total_chunks,
        "result_count": len(result_chunk_ids),
        "result_chunk_ids": result_chunk_ids,
    }
    if retrieval_mode == "semantic":
        metadata["retrieval_method"] = retrieval_method
        metadata["embedding_provider"] = embedding_provider
        metadata["embedding_model"] = embedding_model
        metadata["index_path"] = str(index_path) if index_path is not None else None
    steps = _build_search_steps(
        query=query,
        limit=limit,
        chunks_path=chunks_path,
        total_chunks=total_chunks,
        result_chunk_ids=result_chunk_ids,
        retrieval_mode=retrieval_mode,
        retrieval_method=retrieval_method,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        index_path=index_path,
    )
    return OperationTrace(
        trace_id=f"search-{started_at_utc.strftime('%Y%m%dT%H%M%SZ')}",
        operation="search",
        status="success",
        started_at=_format_timestamp(started_at_utc),
        finished_at=_format_timestamp(finished_at_utc),
        steps=steps,
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
    retrieval_mode: str = "lexical",
    retrieval_method: str = "lexical_token_overlap",
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    index_path: Path | None = None,
) -> OperationTrace:
    started_at_utc = _as_utc(started_at)
    finished_at_utc = _as_utc(finished_at)
    generation_status = (
        "generated"
        if generation_result.status == "success"
        else "not_implemented"
    )
    metadata = {
        "question": question,
        "limit": limit,
        "retrieval_mode": retrieval_mode,
        "retrieval_method": retrieval_method,
        "chunks_path": str(chunks_path),
        "total_chunks": total_chunks,
        "retrieved_count": len(retrieved_chunk_ids),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "generation_status": generation_status,
        "generation_provider": generation_result.provider_name,
        "generation_result_status": generation_result.status,
        "answer_generated": generation_result.answer is not None,
        "config_generation_provider": config_generation_provider,
        "context_chunk_count": context_chunk_count,
        "total_context_chars": total_context_chars,
        "prompt_preview_shown": prompt_preview_shown,
        "max_context_chars": max_context_chars,
    }
    if retrieval_mode == "semantic":
        metadata["embedding_provider"] = embedding_provider
        metadata["embedding_model"] = embedding_model
        metadata["index_path"] = str(index_path) if index_path is not None else None
    metadata.update(_generation_metadata(generation_result.metadata))
    if generation_result.status == "success":
        metadata["source_count"] = len(retrieved_chunk_ids)
    steps = _build_ask_retrieval_steps(
        question=question,
        limit=limit,
        chunks_path=chunks_path,
        total_chunks=total_chunks,
        retrieved_chunk_ids=retrieved_chunk_ids,
        generation_result=generation_result,
        generation_status=generation_status,
        retrieval_mode=retrieval_mode,
        retrieval_method=retrieval_method,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        index_path=index_path,
    )
    return OperationTrace(
        trace_id=f"ask-retrieval-{started_at_utc.strftime('%Y%m%dT%H%M%SZ')}",
        operation="ask_retrieval",
        status="success",
        started_at=_format_timestamp(started_at_utc),
        finished_at=_format_timestamp(finished_at_utc),
        steps=steps,
        metadata=metadata,
    )


def build_index_build_trace(
    embedding_provider: str,
    embedding_model: str,
    chunk_count: int,
    embedding_dim: int,
    index_path: Path,
    chunks_path: Path,
    batch_size: int,
    started_at: datetime,
    finished_at: datetime,
) -> OperationTrace:
    started_at_utc = _as_utc(started_at)
    finished_at_utc = _as_utc(finished_at)
    metadata = {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "chunk_count": chunk_count,
        "embedding_dim": embedding_dim,
        "index_path": str(index_path),
        "chunks_path": str(chunks_path),
        "batch_size": batch_size,
    }
    return OperationTrace(
        trace_id=f"index-build-{started_at_utc.strftime('%Y%m%dT%H%M%SZ')}",
        operation="index_build",
        status="success",
        started_at=_format_timestamp(started_at_utc),
        finished_at=_format_timestamp(finished_at_utc),
        steps=[
            TraceStep(
                name="read_chunks",
                description="Read chunk records from the local workspace.",
                inputs={"chunks_path": str(chunks_path)},
                outputs={"chunk_count": chunk_count},
            ),
            TraceStep(
                name="embed_chunks",
                description="Embed chunk text with the configured embedding provider.",
                inputs={
                    "embedding_provider": embedding_provider,
                    "embedding_model": embedding_model,
                    "batch_size": batch_size,
                },
                outputs={
                    "chunk_count": chunk_count,
                    "embedding_dim": embedding_dim,
                },
            ),
            TraceStep(
                name="write_vector_index",
                description="Persist vector index records to the local workspace.",
                inputs={"chunk_count": chunk_count},
                outputs={"index_path": str(index_path)},
            ),
        ],
        metadata=metadata,
    )


def _build_search_steps(
    query: str,
    limit: int,
    chunks_path: Path,
    total_chunks: int,
    result_chunk_ids: list[str],
    retrieval_mode: str,
    retrieval_method: str,
    embedding_provider: str | None,
    embedding_model: str | None,
    index_path: Path | None,
) -> list[TraceStep]:
    read_chunks_step = TraceStep(
        name="read_chunks",
        description="Read chunk records from the local workspace.",
        inputs={"chunks_path": str(chunks_path)},
        outputs={"total_chunks": total_chunks},
    )
    rank_results_step = TraceStep(
        name="rank_results",
        description=_search_rank_description(retrieval_mode),
        inputs={"matched_chunks": len(result_chunk_ids), "limit": limit},
        outputs={"result_chunk_ids": result_chunk_ids},
    )
    render_results_step = TraceStep(
        name="render_results",
        description="Render search results in the CLI.",
        inputs={"result_count": len(result_chunk_ids)},
        outputs={"status": "success"},
    )

    if retrieval_mode == "semantic":
        return [
            read_chunks_step,
            TraceStep(
                name="embed_query",
                description=(
                    "Embed the search query with the configured embedding provider."
                ),
                inputs={
                    "query": query,
                    "embedding_provider": embedding_provider,
                    "embedding_model": embedding_model,
                },
                outputs={"query_embedding": "computed"},
            ),
            TraceStep(
                name="load_vector_index",
                description="Load local semantic vector index records.",
                inputs={"index_path": str(index_path) if index_path else ""},
                outputs={"status": "loaded"},
            ),
            TraceStep(
                name="score_vectors",
                description="Score indexed chunk vectors by cosine similarity.",
                inputs={"retrieval_method": retrieval_method},
                outputs={"matched_chunks": len(result_chunk_ids)},
            ),
            rank_results_step,
            render_results_step,
        ]

    return [
        read_chunks_step,
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
        rank_results_step,
        render_results_step,
    ]


def _build_ask_retrieval_steps(
    question: str,
    limit: int,
    chunks_path: Path,
    total_chunks: int,
    retrieved_chunk_ids: list[str],
    generation_result: GenerationResult,
    generation_status: str,
    retrieval_mode: str,
    retrieval_method: str,
    embedding_provider: str | None,
    embedding_model: str | None,
    index_path: Path | None,
) -> list[TraceStep]:
    return [
        TraceStep(
            name="read_chunks",
            description="Read chunk records from the local workspace.",
            inputs={"chunks_path": str(chunks_path)},
            outputs={"total_chunks": total_chunks},
        ),
        TraceStep(
            name="retrieve_context",
            description=_ask_retrieval_description(retrieval_mode),
            inputs=_ask_retrieval_inputs(
                question=question,
                limit=limit,
                retrieval_mode=retrieval_mode,
                retrieval_method=retrieval_method,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                index_path=index_path,
            ),
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
                "Run the configured generation provider when retrieved "
                "context is available."
            ),
            inputs={"provider": generation_result.provider_name},
            outputs={
                "generation_status": generation_status,
                "generation_result_status": generation_result.status,
                "answer_generated": generation_result.answer is not None,
            },
        ),
        TraceStep(
            name="render_retrieval_preview",
            description="Render the ask result preview in the CLI.",
            inputs={"retrieved_count": len(retrieved_chunk_ids)},
            outputs={"status": "success"},
        ),
    ]


def _search_rank_description(retrieval_mode: str) -> str:
    if retrieval_mode == "semantic":
        return "Sort semantic search results by score and deterministic chunk id."
    return "Sort results by score and deterministic chunk id."


def _ask_retrieval_description(retrieval_mode: str) -> str:
    if retrieval_mode == "semantic":
        return "Retrieve context chunks with semantic vector search."
    return "Retrieve context chunks with lexical search."


def _ask_retrieval_inputs(
    question: str,
    limit: int,
    retrieval_mode: str,
    retrieval_method: str,
    embedding_provider: str | None,
    embedding_model: str | None,
    index_path: Path | None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {"question": question, "limit": limit}
    if retrieval_mode == "semantic":
        inputs.update(
            {
                "retrieval_method": retrieval_method,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "index_path": str(index_path) if index_path is not None else None,
            }
        )
    return inputs


def _generation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {"model", "base_url", "endpoint", "skip_reason"}
    return {key: value for key, value in metadata.items() if key in allowed_keys}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")
