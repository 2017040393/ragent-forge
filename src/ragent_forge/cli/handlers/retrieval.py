from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from rich.console import Console

from ragent_forge.app.models import AppConfig, ContextPack
from ragent_forge.app.services.ask_service import AskService
from ragent_forge.app.services.chunk_service import make_preview
from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.context_service import build_generation_prompt
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.services.trace_service import (
    build_ask_retrieval_trace,
    build_search_trace,
)
from ragent_forge.app.source_labels import format_source_label, format_source_range
from ragent_forge.cli.handlers.chunks import _print_no_chunks
from ragent_forge.composition import (
    RetrievalRuntime,
    build_generation_service,
    build_retrieval_runtime,
)
from ragent_forge.core.retrieval.types import RETRIEVAL_MODES, RetrievalMode
from ragent_forge.infrastructure.local_workspace import LocalWorkspace

RETRIEVAL_CHOICES = list(RETRIEVAL_MODES)

BuiltSearchService = RetrievalRuntime


def _build_search_service_for_retrieval(
    workspace: LocalWorkspace,
    retrieval: RetrievalMode,
    limit: int,
    config: AppConfig | None = None,
) -> BuiltSearchService:
    return build_retrieval_runtime(workspace, retrieval, limit=limit, config=config)


def _as_retrieval_mode(retrieval: str) -> RetrievalMode:
    if retrieval not in RETRIEVAL_CHOICES:
        raise ValueError(f"Unsupported retrieval mode: {retrieval}")
    return cast(RetrievalMode, retrieval)


def _handle_search(
    console: Console, workspace_path: str, query: str, limit: int, retrieval: str
) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0
    started_at = datetime.now(UTC)
    try:
        if retrieval in {"semantic", "hybrid"} and (not workspace.has_vector_index()):
            failure_label = (
                "Hybrid search failed"
                if retrieval == "hybrid"
                else "Semantic search failed"
            )
            console.print(
                f"{failure_label}: vector index not found. "
                "Run `ragent index build` first.",
                markup=False,
                soft_wrap=True,
            )
            return 1
        built_search = _build_search_service_for_retrieval(
            workspace, _as_retrieval_mode(retrieval), limit
        )
        retrieval_run = built_search.retrieval_engine.run(query, limit)
        results = retrieval_run.results
        total_chunks = built_search.search_service.count_chunks()
    except (OSError, RuntimeError, ValueError) as exc:
        failure_label = (
            "Hybrid search failed"
            if retrieval == "hybrid"
            else "Semantic search failed"
            if retrieval == "semantic"
            else "Search failed"
        )
        console.print(f"{failure_label}: {exc}", markup=False, soft_wrap=True)
        return 1
    finished_at = datetime.now(UTC)
    trace = build_search_trace(
        query=query,
        limit=limit,
        chunks_path=workspace.chunks_path,
        total_chunks=total_chunks,
        result_chunk_ids=[result.chunk_id for result in results],
        started_at=started_at,
        finished_at=finished_at,
        retrieval_mode=retrieval,
        retrieval_method=built_search.retrieval_method,
        fusion_method=built_search.fusion_method,
        rrf_k=built_search.rrf_k,
        sparse_method=built_search.sparse_method,
        dense_method=built_search.dense_method,
        sparse_weight=built_search.sparse_weight,
        dense_weight=built_search.dense_weight,
        lexical_weight=built_search.lexical_weight,
        semantic_weight=built_search.semantic_weight,
        candidate_limit=built_search.candidate_limit,
        embedding_provider=built_search.embedding_provider,
        embedding_model=built_search.embedding_model,
        index_path=built_search.index_path,
        retrieval_run=retrieval_run,
    )
    trace_path = workspace.write_trace(trace)
    console.print(f"Search query: {query}")
    console.print(f"Retrieval mode: {retrieval}")
    if not results:
        console.print("No matches found.")
        console.print(f"Saved trace to: {trace_path}")
        return 0
    console.print(f"Results: {len(results)}")
    console.print()
    for index, result in enumerate(results, start=1):
        range_text = _format_search_range(
            result.start_char, result.end_char, result.metadata
        )
        console.print(
            f"{index}. score={result.score:g} | {result.chunk_id}", soft_wrap=True
        )
        console.print(
            f"Source: {format_source_label(result.source_path, result.metadata)}",
            soft_wrap=True,
        )
        console.print(f"Range: {range_text}")
        console.print(f"Preview: {make_preview(result.text)}")
        console.print()
    console.print(f"Saved trace to: {trace_path}")
    return 0


def _format_search_range(
    start_char: int | None,
    end_char: int | None,
    metadata: dict[str, object] | None = None,
) -> str:
    return format_source_range(start_char, end_char, metadata)


def _handle_ask(
    console: Console,
    workspace_path: str,
    question: str,
    limit: int,
    show_prompt: bool,
    max_context_chars: int,
    retrieval: str,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0
    started_at = datetime.now(UTC)
    try:
        config = ConfigService(workspace).load()
        generation_service = build_generation_service(config)
        if retrieval in {"semantic", "hybrid"} and (not workspace.has_vector_index()):
            console.print(
                "Ask failed: vector index not found. Run `ragent index build` first.",
                markup=False,
                soft_wrap=True,
            )
            return 1
        built_search = _build_search_service_for_retrieval(
            workspace, _as_retrieval_mode(retrieval), limit, config=config
        )
        ask_service = AskService(
            workspace,
            generation_service=generation_service,
            search_service=built_search.retrieval_engine,
            retrieval_engine=built_search.retrieval_engine,
            retrieval_method=built_search.retrieval_method,
        )
        result = ask_service.ask(question, limit, max_context_chars)
        total_chunks = ask_service.count_chunks()
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Ask failed:[/bold red] {exc}")
        return 1
    finished_at = datetime.now(UTC)
    retrieved_chunk_ids = [search_result.chunk_id for search_result in result.results]
    trace = build_ask_retrieval_trace(
        question=question,
        limit=limit,
        chunks_path=workspace.chunks_path,
        total_chunks=total_chunks,
        retrieved_chunk_ids=retrieved_chunk_ids,
        generation_result=result.generation_result,
        config_generation_provider=config.generation.provider,
        context_chunk_count=len(result.context_pack.context_chunks),
        total_context_chars=result.context_pack.total_context_chars,
        prompt_preview_shown=show_prompt,
        max_context_chars=max_context_chars,
        started_at=started_at,
        finished_at=finished_at,
        retrieval_mode=retrieval,
        retrieval_method=built_search.retrieval_method,
        fusion_method=built_search.fusion_method,
        rrf_k=built_search.rrf_k,
        sparse_method=built_search.sparse_method,
        dense_method=built_search.dense_method,
        sparse_weight=built_search.sparse_weight,
        dense_weight=built_search.dense_weight,
        lexical_weight=built_search.lexical_weight,
        semantic_weight=built_search.semantic_weight,
        embedding_provider=built_search.embedding_provider,
        embedding_model=built_search.embedding_model,
        index_path=built_search.index_path,
        retrieval_run=result.retrieval_run,
    )
    trace_path = workspace.write_trace(trace)
    if result.generation_result.status == "success":
        console.print("Ask pipeline: generated answer mode")
    else:
        console.print("Ask pipeline: retrieval-only mode")
    console.print()
    console.print(f"Question: {question}")
    console.print(f"Retrieval mode: {retrieval}")
    if result.generation_result.status == "success":
        console.print(f"Generation provider: {result.generation_result.provider_name}")
        console.print(f"Generation status: {result.generation_result.status}")
    elif result.generation_result.provider_name == "null":
        console.print("Generation: not configured.")
    console.print()
    if result.generation_result.status == "success":
        console.print("Answer:")
        console.print(result.answer or "")
        console.print()
        console.print("Sources:")
        for index, search_result in enumerate(result.results, start=1):
            _print_source_line(console, index, search_result)
        console.print()
        if show_prompt:
            _print_context_pack(console, result.context_pack)
    elif result.generation_result.status == "skipped":
        console.print("No retrieved context found.")
        console.print("Generation skipped because there is no retrieved context.")
        console.print()
        if show_prompt:
            _print_context_pack(console, result.context_pack)
    else:
        if result.results:
            console.print("Retrieved context:")
            for index, search_result in enumerate(result.results, start=1):
                _print_retrieved_context(console, index, search_result)
            console.print()
        else:
            console.print("No retrieved context found.")
            console.print()
        if show_prompt:
            _print_context_pack(console, result.context_pack)
    console.print(f"Saved trace to: {trace_path}")
    return 0


def _print_retrieved_context(
    console: Console, index: int, search_result: SearchResult
) -> None:
    range_text = _format_search_range(
        search_result.start_char, search_result.end_char, search_result.metadata
    )
    console.print(
        f"{index}. score={search_result.score:g} | {search_result.chunk_id}",
        soft_wrap=True,
    )
    console.print(
        "Source: "
        f"{format_source_label(search_result.source_path, search_result.metadata)}",
        soft_wrap=True,
    )
    console.print(f"Range: {range_text}")
    console.print(f"Preview: {make_preview(search_result.text)}")


def _print_source_line(
    console: Console, index: int, search_result: SearchResult
) -> None:
    range_text = _format_search_range(
        search_result.start_char, search_result.end_char, search_result.metadata
    )
    console.print(f"{index}. {search_result.chunk_id}")
    console.print(
        "   Source: "
        f"{format_source_label(search_result.source_path, search_result.metadata)}"
    )
    console.print(f"   Range: {range_text}")
    console.print(f"   Score: {search_result.score:g}")


def _print_context_pack(console: Console, context_pack: ContextPack) -> None:
    console.print("Context pack:")
    console.print(f"Context chunks: {len(context_pack.context_chunks)}")
    console.print(f"Total context chars: {context_pack.total_context_chars}")
    console.print(f"Retrieval method: {context_pack.retrieval_method}")
    console.print()
    console.print("Generation prompt:")
    console.print(build_generation_prompt(context_pack), soft_wrap=True)
    console.print()
