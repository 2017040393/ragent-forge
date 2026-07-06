from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from rich.console import Console

from ragent_forge.app.models import (
    AppConfig,
    ContextPack,
    WorkspaceStatus,
)
from ragent_forge.app.services.ask_service import (
    AskService,
)
from ragent_forge.app.services.ask_service import (
    SearchServiceProtocol as AskSearchServiceProtocol,
)
from ragent_forge.app.services.chunk_service import ChunkService, make_preview
from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.context_service import build_generation_prompt
from ragent_forge.app.services.embedding_service import EmbeddingService
from ragent_forge.app.services.eval_dataset_generation_service import (
    EvalDatasetGenerationReport,
    EvalDatasetGenerationService,
    TextGenerationClient,
    write_jsonl,
)
from ragent_forge.app.services.evidence_span_service import EvidenceSpanService
from ragent_forge.app.services.generation_service import GenerationService
from ragent_forge.app.services.hybrid_search_service import (
    HybridSearchConfig,
    HybridSearchService,
)
from ragent_forge.app.services.index_service import IndexBuildService
from ragent_forge.app.services.ingest_service import IngestService
from ragent_forge.app.services.retrieval_eval_service import (
    RetrievalEvalReport,
    RetrievalEvalService,
)
from ragent_forge.app.services.search_service import LexicalSearchService, SearchResult
from ragent_forge.app.services.semantic_search_service import SemanticSearchService
from ragent_forge.app.services.text_generation_client import (
    OpenAIResponsesTextGenerationClient,
)
from ragent_forge.app.services.trace_history_service import TraceHistoryService
from ragent_forge.app.services.trace_service import (
    build_ask_retrieval_trace,
    build_index_build_trace,
    build_ingest_trace,
    build_retrieval_eval_trace,
    build_search_trace,
)
from ragent_forge.app.services.vector_index_service import VectorIndexService
from ragent_forge.app.source_labels import format_source_label, format_source_range
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.tui.main import RagentForgeApp

RETRIEVAL_CHOICES = ["lexical", "semantic", "hybrid"]
RetrievalMode = Literal["lexical", "semantic", "hybrid"]


@dataclass(frozen=True)
class BuiltSearchService:
    search_service: AskSearchServiceProtocol
    retrieval_method: str
    embedding_provider: str | None = None
    embedding_model: str | None = None
    index_path: Path | None = None
    fusion_method: str | None = None
    rrf_k: int | None = None
    lexical_weight: float | None = None
    semantic_weight: float | None = None
    candidate_limit: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragent",
        description=(
            "A local-first TUI workbench for inspectable Agentic RAG workflows "
            "over personal knowledge bases."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("tui", help="Launch the local Textual TUI.")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest local Markdown/TXT/PDF knowledge folders or files.",
    )
    ingest_parser.add_argument("path", help="Path to a local knowledge folder or file.")
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk.",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Characters to overlap between adjacent chunks.",
    )
    ingest_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory for generated ingestion data.",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show local RAGentForge workspace status.",
    )
    status_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )

    config_parser = subparsers.add_parser(
        "config",
        help="Inspect or initialize local RAGentForge configuration.",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    config_show_parser = config_subparsers.add_parser(
        "show",
        help="Show the effective local configuration.",
    )
    config_show_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )

    config_init_parser = config_subparsers.add_parser(
        "init",
        help="Write the default local configuration.",
    )
    config_init_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to initialize.",
    )
    config_init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing local configuration.",
    )

    chunks_parser = subparsers.add_parser(
        "chunks",
        help="Inspect generated chunks in a local workspace.",
    )
    chunks_subparsers = chunks_parser.add_subparsers(dest="chunks_command")

    chunks_list_parser = chunks_subparsers.add_parser(
        "list",
        help="List generated chunks.",
    )
    chunks_list_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    chunks_list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of chunks to show.",
    )

    chunks_show_parser = chunks_subparsers.add_parser(
        "show",
        help="Show a generated chunk by exact chunk id.",
    )
    chunks_show_parser.add_argument("chunk_id", help="Exact chunk id to inspect.")
    chunks_show_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )

    traces_parser = subparsers.add_parser(
        "traces",
        help="Inspect local operation traces.",
    )
    traces_subparsers = traces_parser.add_subparsers(dest="traces_command")
    traces_latest_parser = traces_subparsers.add_parser(
        "latest",
        help="Show the latest local operation trace.",
    )
    traces_latest_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    traces_list_parser = traces_subparsers.add_parser(
        "list",
        help="List local operation trace history.",
    )
    traces_list_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    traces_list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of traces to show.",
    )
    traces_show_parser = traces_subparsers.add_parser(
        "show",
        help="Show a local operation trace by exact trace id.",
    )
    traces_show_parser.add_argument("trace_id", help="Exact trace id to inspect.")
    traces_show_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run local evaluation workflows.",
    )
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    retrieval_eval_parser = eval_subparsers.add_parser(
        "retrieval",
        help=(
            "Evaluate lexical, semantic, or hybrid retrieval quality "
            "with JSONL cases."
        ),
    )
    retrieval_eval_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to evaluate.",
    )
    retrieval_eval_parser.add_argument(
        "--cases",
        required=True,
        help="Path to a JSONL retrieval eval cases file.",
    )
    retrieval_eval_parser.add_argument(
        "--retrieval",
        choices=RETRIEVAL_CHOICES,
        default="lexical",
        help="Retrieval mode to evaluate.",
    )
    retrieval_eval_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of top retrieval results to evaluate.",
    )
    retrieval_eval_parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path for the JSON retrieval eval report.",
    )
    eval_generate_parser = eval_subparsers.add_parser(
        "generate",
        help="Generate span-based synthetic retrieval eval JSONL cases.",
    )
    eval_generate_parser.add_argument(
        "--source",
        required=True,
        help="Path to a source document file or directory.",
    )
    eval_generate_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory for configuration.",
    )
    eval_generate_parser.add_argument(
        "--output",
        required=True,
        help="Path to write generated JSONL eval cases.",
    )
    eval_generate_parser.add_argument(
        "--questions-per-span",
        type=int,
        default=2,
        help="Synthetic questions to generate for each evidence span.",
    )
    eval_generate_parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of generated eval cases.",
    )
    eval_generate_parser.add_argument(
        "--min-evidence-chars",
        type=int,
        default=250,
        help="Minimum evidence span characters.",
    )
    eval_generate_parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=1200,
        help="Maximum evidence span characters.",
    )
    eval_generate_parser.add_argument(
        "--include-pdf",
        action="store_true",
        help="Opt in to experimental PDF evidence span extraction.",
    )
    eval_generate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output JSONL file.",
    )
    eval_generate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract spans and print counts without calling the LLM.",
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Build or inspect the local semantic vector index.",
    )
    index_subparsers = index_parser.add_subparsers(dest="index_command")

    index_build_parser = index_subparsers.add_parser(
        "build",
        help="Build the local semantic vector index from generated chunks.",
    )
    index_build_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to index.",
    )

    index_status_parser = index_subparsers.add_parser(
        "status",
        help="Show local semantic vector index status.",
    )
    index_status_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search generated chunks with lexical, semantic, or hybrid retrieval.",
    )
    search_parser.add_argument("query", help="Query to search for.")
    search_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to search.",
    )
    search_parser.add_argument(
        "--retrieval",
        choices=RETRIEVAL_CHOICES,
        default="lexical",
        help="Retrieval mode to use.",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to show.",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question using local retrieval and optional generation.",
    )
    ask_parser.add_argument("question", help="Question to ask.")
    ask_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to search.",
    )
    ask_parser.add_argument(
        "--retrieval",
        choices=RETRIEVAL_CHOICES,
        default="lexical",
        help="Retrieval mode to use.",
    )
    ask_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of context chunks to show.",
    )
    ask_parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Show the deterministic local prompt preview.",
    )
    ask_parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Maximum retrieved context characters in the prompt preview.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.command is None or args.command == "tui":
        RagentForgeApp().run()
        return 0

    if args.command == "ingest":
        started_at = datetime.now(UTC)
        try:
            result = IngestService(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            ).ingest(args.path)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[bold red]Ingest failed:[/bold red] {exc}")
            return 1

        workspace = LocalWorkspace(args.workspace)
        chunks_path = workspace.write_chunks(result.chunks)
        summary_path = workspace.write_ingest_summary(result)
        finished_at = datetime.now(UTC)
        trace = build_ingest_trace(
            result=result,
            chunks_path=chunks_path,
            summary_path=summary_path,
            started_at=started_at,
            finished_at=finished_at,
        )
        trace_path = workspace.write_trace(trace)

        console.print("[bold green]Ingest complete[/bold green]")
        console.print(f"Source: [cyan]{result.source_path}[/cyan]")
        console.print(f"Documents: {result.document_count}")
        console.print(f"Chunks: {result.chunk_count}")
        console.print(f"Skipped files: {result.skipped_count}")
        console.print(f"Chunk size: {result.metadata['chunk_size']}")
        console.print(f"Chunk overlap: {result.metadata['chunk_overlap']}")
        console.print(f"Saved chunks to: {chunks_path}")
        console.print(f"Saved summary to: {summary_path}")
        console.print(f"Saved trace to: {trace_path}")
        return 0

    if args.command == "status":
        workspace = LocalWorkspace(args.workspace)
        try:
            workspace_status = workspace.status()
        except ValueError as exc:
            console.print(f"[bold red]Status failed:[/bold red] {exc}")
            return 1

        _print_workspace_status(console, workspace_status)
        return 0

    if args.command == "chunks":
        if args.chunks_command == "list":
            return _handle_chunks_list(console, args.workspace, args.limit)
        if args.chunks_command == "show":
            return _handle_chunks_show(console, args.workspace, args.chunk_id)
        parser.print_help()
        return 0

    if args.command == "config":
        if args.config_command == "show":
            return _handle_config_show(console, args.workspace)
        if args.config_command == "init":
            return _handle_config_init(console, args.workspace, args.overwrite)
        parser.print_help()
        return 0

    if args.command == "traces":
        if args.traces_command == "latest":
            return _handle_traces_latest(console, args.workspace)
        if args.traces_command == "list":
            return _handle_traces_list(console, args.workspace, args.limit)
        if args.traces_command == "show":
            return _handle_traces_show(console, args.workspace, args.trace_id)
        parser.print_help()
        return 0

    if args.command == "eval":
        if args.eval_command == "retrieval":
            return _handle_eval_retrieval(
                console,
                args.workspace,
                args.cases,
                args.retrieval,
                args.limit,
                args.report_path,
            )
        if args.eval_command == "generate":
            return _handle_eval_generate(
                console,
                args.workspace,
                args.source,
                args.output,
                args.questions_per_span,
                args.max_cases,
                args.min_evidence_chars,
                args.max_evidence_chars,
                args.include_pdf,
                args.overwrite,
                args.dry_run,
            )
        parser.print_help()
        return 0

    if args.command == "index":
        if args.index_command == "build":
            return _handle_index_build(console, args.workspace)
        if args.index_command == "status":
            return _handle_index_status(console, args.workspace)
        parser.print_help()
        return 0

    if args.command == "search":
        return _handle_search(
            console,
            args.workspace,
            args.query,
            args.limit,
            args.retrieval,
        )

    if args.command == "ask":
        return _handle_ask(
            console,
            args.workspace,
            args.question,
            args.limit,
            args.show_prompt,
            args.max_context_chars,
            args.retrieval,
        )

    parser.print_help()
    return 0


def _print_workspace_status(
    console: Console,
    workspace_status: WorkspaceStatus,
) -> None:
    console.print(f"Workspace: [cyan]{workspace_status.root_path}[/cyan]")

    if workspace_status.status == "not_initialized":
        console.print("Status: not initialized")
        console.print()
        console.print("Run `ragent ingest <path>` to create a local workspace.")
        return

    if workspace_status.status == "incomplete":
        console.print("Status: incomplete")
        console.print()
        for missing_file in workspace_status.missing_files:
            if missing_file == workspace_status.latest_summary_path:
                console.print(f"Missing summary file: {missing_file}")
            elif missing_file == workspace_status.chunks_path:
                console.print(f"Missing chunks file: {missing_file}")
            else:
                console.print(f"Missing file: {missing_file}")
        console.print("Run `ragent ingest <path>` to regenerate workspace data.")
        return

    summary = workspace_status.summary
    console.print("Status: ready")
    console.print()
    console.print(f"Last ingest source: {summary.get('source_path', '')}")
    console.print(f"Documents: {summary.get('document_count', 0)}")
    chunk_count = (
        workspace_status.chunk_count_from_file
        if workspace_status.chunk_count_from_file is not None
        else summary.get("chunk_count", 0)
    )
    console.print(f"Chunks: {chunk_count}")
    console.print(f"Skipped files: {summary.get('skipped_count', 0)}")
    console.print()
    console.print(f"Chunks file: {workspace_status.chunks_path}")
    console.print(f"Summary file: {workspace_status.latest_summary_path}")


def _handle_chunks_list(console: Console, workspace_path: str, limit: int) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0

    service = ChunkService(workspace)
    try:
        chunks = service.list_chunks(limit)
        total_count = service.count_chunks()
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Chunks failed:[/bold red] {exc}")
        return 1

    console.print("Chunks")
    console.print("Chunk ID | Source | Range | Preview")

    for chunk in chunks:
        metadata = chunk.get("metadata")
        source_label = format_source_label(
            str(chunk.get("source_path", "")),
            metadata if isinstance(metadata, dict) else None,
        )
        console.print(
            f"{chunk.get('chunk_id', '')} | "
            f"{source_label} | "
            f"{_format_char_range(chunk)} | "
            f"{make_preview(str(chunk.get('text', '')))}",
            soft_wrap=True,
        )

    if total_count > limit:
        console.print(
            f"Showing {len(chunks)} of {total_count} chunks. Use --limit to show more."
        )
    return 0


def _handle_chunks_show(
    console: Console,
    workspace_path: str,
    chunk_id: str,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0

    try:
        chunk = ChunkService(workspace).get_chunk(chunk_id)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Chunks failed:[/bold red] {exc}")
        return 1

    if chunk is None:
        console.print(f"Chunk not found: {chunk_id}")
        return 0

    console.print(f"[bold]Chunk ID:[/bold] {chunk.get('chunk_id', '')}", soft_wrap=True)
    console.print(
        f"[bold]Document ID:[/bold] {chunk.get('document_id', '')}",
        soft_wrap=True,
    )
    console.print(
        f"[bold]Source path:[/bold] {chunk.get('source_path', '')}",
        soft_wrap=True,
    )
    console.print(f"[bold]Start char:[/bold] {chunk.get('start_char', '')}")
    console.print(f"[bold]End char:[/bold] {chunk.get('end_char', '')}")
    console.print()
    console.print("[bold]Metadata:[/bold]")
    console.print(json.dumps(chunk.get("metadata", {}), ensure_ascii=False, indent=2))
    console.print()
    console.print("[bold]Text:[/bold]")
    console.print(str(chunk.get("text", "")))
    return 0


def _format_char_range(chunk: dict[str, object]) -> str:
    start_char = chunk.get("start_char")
    end_char = chunk.get("end_char")
    metadata = chunk.get("metadata")
    return format_source_range(
        start_char if isinstance(start_char, int) else None,
        end_char if isinstance(end_char, int) else None,
        metadata if isinstance(metadata, dict) else None,
    )


def _print_no_chunks(console: Console) -> None:
    console.print("No chunks found. Run ragent ingest <path> first.")


def _handle_config_show(console: Console, workspace_path: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    config_service = ConfigService(workspace)
    try:
        config = config_service.load()
    except ValueError as exc:
        console.print(f"[bold red]Config failed:[/bold red] {exc}")
        return 1

    if workspace.config_path.is_file():
        console.print(f"Config: {workspace.config_path}", soft_wrap=True)
    else:
        console.print("Config: default")
    console.print()
    console.print(f"generation.provider: {config.generation.provider}")
    if config.generation.provider == "openai_responses":
        console.print(f"generation.base_url: {config.generation.base_url}")
        console.print(f"generation.model: {config.generation.model}")
        if config.generation.api_key:
            console.print("generation.api_key: <hidden>")
        console.print(
            f"generation.timeout_seconds: {config.generation.timeout_seconds}"
        )
        console.print(f"generation.temperature: {config.generation.temperature}")
        if config.generation.reasoning_effort is not None:
            console.print(
                f"generation.reasoning_effort: {config.generation.reasoning_effort}"
            )
    console.print(f"embedding.provider: {config.embedding.provider}")
    if config.embedding.provider == "openai_embeddings":
        console.print(f"embedding.base_url: {config.embedding.base_url}")
        console.print(f"embedding.model: {config.embedding.model}")
        if config.embedding.api_key:
            console.print("embedding.api_key: <hidden>")
        console.print(f"embedding.timeout_seconds: {config.embedding.timeout_seconds}")
        console.print(f"embedding.batch_size: {config.embedding.batch_size}")
    return 0


def _handle_config_init(
    console: Console,
    workspace_path: str,
    overwrite: bool,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    config_service = ConfigService(workspace)
    config_exists = workspace.config_path.exists()
    config_path = config_service.write_default(overwrite=overwrite)

    if config_exists and not overwrite:
        console.print(f"Config already exists: {config_path}", soft_wrap=True)
        return 0

    console.print(f"Wrote default config to: {config_path}", soft_wrap=True)
    return 0


def _handle_traces_latest(console: Console, workspace_path: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_latest_trace():
        console.print("No trace found. Run ragent ingest <path> first.")
        return 0

    try:
        trace = workspace.read_latest_trace()
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Traces failed:[/bold red] {exc}")
        return 1

    _print_trace(console, trace)
    return 0


def _handle_traces_list(console: Console, workspace_path: str, limit: int) -> int:
    workspace = LocalWorkspace(workspace_path)
    result = TraceHistoryService(workspace).list_traces(limit)

    if result.traces:
        console.print("Traces")
        console.print("Trace ID | Operation | Status | Started at | Finished at")
        for trace in result.traces:
            console.print(
                f"{trace.trace_id} | "
                f"{trace.operation} | "
                f"{trace.status} | "
                f"{trace.started_at} | "
                f"{trace.finished_at or ''}",
                soft_wrap=True,
            )
    elif result.warnings:
        console.print("No valid traces found.")
    else:
        console.print("No traces found. Run ragent ingest <path> first.")

    _print_trace_warnings(console, result.warnings)
    return 0


def _handle_traces_show(
    console: Console,
    workspace_path: str,
    trace_id: str,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    try:
        trace = TraceHistoryService(workspace).read_trace(trace_id)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Traces failed:[/bold red] {exc}")
        return 1

    if trace is None:
        console.print(f"Trace not found: {trace_id}")
        return 0

    _print_trace(console, trace)
    return 0


def _handle_index_build(console: Console, workspace_path: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0

    started_at = datetime.now(UTC)
    try:
        config = ConfigService(workspace).load()
        embedding_service = EmbeddingService.from_config(config)
        result = IndexBuildService(
            workspace,
            embedding_service=embedding_service,
        ).build(
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model,
            batch_size=config.embedding.batch_size,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"Index build failed: {exc}", markup=False, soft_wrap=True)
        return 1
    finished_at = datetime.now(UTC)
    trace = build_index_build_trace(
        embedding_provider=result.embedding_provider,
        embedding_model=result.embedding_model,
        chunk_count=result.chunk_count,
        embedding_dim=result.embedding_dim,
        index_path=result.index_path,
        chunks_path=result.chunks_path,
        batch_size=result.batch_size,
        started_at=started_at,
        finished_at=finished_at,
    )
    trace_path = workspace.write_trace(trace)

    console.print("Semantic index build")
    console.print()
    console.print(f"Embedding provider: {result.embedding_provider}")
    console.print(f"Embedding model: {result.embedding_model}")
    console.print(f"Chunks embedded: {result.chunk_count}")
    console.print(f"Embedding dim: {result.embedding_dim}")
    console.print(f"Index path: {result.index_path}")
    console.print(f"Saved trace to: {trace_path}")
    return 0


def _handle_index_status(console: Console, workspace_path: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_vector_index():
        console.print("Semantic index: missing")
        console.print("Run `ragent index build` to create it.")
        return 0

    try:
        manifest = VectorIndexService(workspace).read_manifest()
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Index status failed:[/bold red] {exc}")
        return 1

    console.print("Semantic index: ready")
    index_path = manifest.get("index_path", workspace.vector_index_path)
    console.print(f"Index path: {index_path}")
    console.print(f"Chunks indexed: {manifest.get('chunk_count', 0)}")
    console.print(f"Embedding model: {manifest.get('embedding_model', '')}")
    console.print(f"Embedding dim: {manifest.get('embedding_dim', 0)}")
    console.print(f"Built at: {manifest.get('built_at', '')}")
    return 0


def _build_search_service_for_retrieval(
    workspace: LocalWorkspace,
    retrieval: RetrievalMode,
    limit: int,
    config=None,
) -> BuiltSearchService:
    if retrieval == "semantic":
        config = config or ConfigService(workspace).load()
        embedding_service = EmbeddingService.from_config(config)
        return BuiltSearchService(
            search_service=SemanticSearchService(workspace, embedding_service),
            retrieval_method="semantic_cosine_similarity",
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model,
            index_path=workspace.vector_index_path,
        )

    if retrieval == "hybrid":
        config = config or ConfigService(workspace).load()
        embedding_service = EmbeddingService.from_config(config)
        semantic_search_service = SemanticSearchService(workspace, embedding_service)
        hybrid_config = HybridSearchConfig()
        hybrid_search_service = HybridSearchService(
            lexical_search_service=LexicalSearchService(workspace),
            semantic_search_service=semantic_search_service,
            config=hybrid_config,
        )
        return BuiltSearchService(
            search_service=hybrid_search_service,
            retrieval_method="hybrid_rrf",
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model,
            index_path=workspace.vector_index_path,
            fusion_method="reciprocal_rank_fusion",
            rrf_k=hybrid_config.rrf_k,
            lexical_weight=hybrid_config.lexical_weight,
            semantic_weight=hybrid_config.semantic_weight,
            candidate_limit=hybrid_search_service.candidate_limit_for(limit),
        )

    return BuiltSearchService(
        search_service=LexicalSearchService(workspace),
        retrieval_method="lexical_token_overlap",
    )


def _as_retrieval_mode(retrieval: str) -> RetrievalMode:
    if retrieval not in RETRIEVAL_CHOICES:
        raise ValueError(f"Unsupported retrieval mode: {retrieval}")
    return cast(RetrievalMode, retrieval)


def _build_text_generation_client(config: AppConfig) -> TextGenerationClient:
    if config.generation.provider == "openai_responses":
        return OpenAIResponsesTextGenerationClient.from_config(config)
    raise ValueError(f"Unsupported generation provider: {config.generation.provider}")


def _handle_eval_generate(
    console: Console,
    workspace_path: str,
    source_path: str,
    output_path: str,
    questions_per_span: int,
    max_cases: int | None,
    min_evidence_chars: int,
    max_evidence_chars: int,
    include_pdf: bool,
    overwrite: bool,
    dry_run: bool,
) -> int:
    try:
        if questions_per_span < 1:
            raise ValueError("questions_per_span must be greater than 0")
        if max_cases is not None and max_cases < 0:
            raise ValueError("max_cases must be greater than or equal to 0")
        output = Path(output_path)
        if output.exists() and not overwrite and not dry_run:
            raise FileExistsError(f"Output JSONL already exists: {output}")

        workspace = LocalWorkspace(workspace_path)
        config = ConfigService(workspace).load()
        if config.generation.provider == "null" and not dry_run:
            console.print(
                "Eval generation failed: generation provider is not configured. "
                "Set generation.provider to openai_responses or use --dry-run.",
                markup=False,
                soft_wrap=True,
            )
            return 1

        spans = EvidenceSpanService(
            min_chars=min_evidence_chars,
            max_chars=max_evidence_chars,
            include_pdf=include_pdf,
        ).extract(source_path)
        if not spans:
            console.print(
                "Eval generation failed: "
                f"no evidence spans extracted from {source_path}",
                markup=False,
                soft_wrap=True,
            )
            return 1

        if dry_run:
            _print_eval_generate_dry_run_summary(
                console=console,
                source_path=source_path,
                span_count=len(spans),
                include_pdf=include_pdf,
                min_evidence_chars=min_evidence_chars,
                max_evidence_chars=max_evidence_chars,
                questions_per_span=questions_per_span,
                max_cases=max_cases,
            )
            return 0

        text_generation_client = _build_text_generation_client(config)
        report = EvalDatasetGenerationService(
            generator=text_generation_client,
            questions_per_span=questions_per_span,
        ).generate(spans, max_cases=max_cases)
        if report.generated_count == 0:
            _print_eval_generate_empty_failure(console, report)
            return 1
        written_output_path = write_jsonl(
            report.cases,
            output,
            overwrite=overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        console.print(
            f"Eval generation failed: {exc}",
            markup=False,
            soft_wrap=True,
        )
        return 1

    _print_eval_generate_summary(
        console=console,
        source_path=source_path,
        output_path=written_output_path,
        span_count=len(spans),
        report=report,
        workspace_path=workspace_path,
    )
    return 0


def _print_eval_generate_dry_run_summary(
    *,
    console: Console,
    source_path: str,
    span_count: int,
    include_pdf: bool,
    min_evidence_chars: int,
    max_evidence_chars: int,
    questions_per_span: int,
    max_cases: int | None,
) -> None:
    estimated_cases = span_count * questions_per_span
    if max_cases is not None:
        estimated_cases = min(estimated_cases, max_cases)
    max_cases_text = str(max_cases) if max_cases is not None else "none"

    console.print("Eval dataset generation dry run")
    console.print(f"Source: {source_path}", soft_wrap=True)
    console.print(f"Evidence spans extracted: {span_count}")
    console.print(f"include_pdf: {include_pdf}")
    console.print(f"min_evidence_chars: {min_evidence_chars}")
    console.print(f"max_evidence_chars: {max_evidence_chars}")
    console.print(f"questions_per_span: {questions_per_span}")
    console.print(f"max_cases: {max_cases_text}")
    console.print(f"Estimated max generated cases: {estimated_cases}")


def _print_eval_generate_empty_failure(
    console: Console,
    report: EvalDatasetGenerationReport,
) -> None:
    console.print("Eval generation failed: no eval cases were generated.")
    console.print(f"Spans skipped: {report.skipped_count}")
    console.print(f"Error count: {len(report.errors)}")
    if not report.errors:
        return
    console.print("Errors:")
    for error in report.errors[:5]:
        span_id = str(error.get("span_id", ""))
        message = str(error.get("message", ""))
        console.print(f"- {span_id}: {message}", soft_wrap=True)


def _print_eval_generate_summary(
    *,
    console: Console,
    source_path: str,
    output_path: Path,
    span_count: int,
    report: EvalDatasetGenerationReport,
    workspace_path: str,
) -> None:
    console.print("Eval dataset generation")
    console.print(f"Source: {source_path}", soft_wrap=True)
    console.print(f"Output: {output_path}", soft_wrap=True)
    console.print(f"Evidence spans extracted: {span_count}")
    console.print(f"Cases generated: {report.generated_count}")
    console.print(f"Spans skipped: {report.skipped_count}")
    console.print(f"Error count: {len(report.errors)}")
    console.print(f"Generation method: {report.metadata['generation_method']}")

    if report.errors:
        console.print("Errors:")
        for error in report.errors[:5]:
            span_id = str(error.get("span_id", ""))
            message = str(error.get("message", ""))
            console.print(f"- {span_id}: {message}", soft_wrap=True)

    console.print()
    console.print(
        "Next: "
        f"ragent eval retrieval --cases {output_path} --workspace {workspace_path} "
        "--retrieval lexical --limit 5",
        soft_wrap=True,
    )


def _handle_eval_retrieval(
    console: Console,
    workspace_path: str,
    cases_path: str,
    retrieval: str,
    limit: int,
    report_path: str | None,
) -> int:
    started_at = datetime.now(UTC)
    workspace = LocalWorkspace(workspace_path)
    eval_service = RetrievalEvalService()

    try:
        if limit < 1:
            raise ValueError("limit must be greater than 0")
        cases = eval_service.load_cases(cases_path)
        if not workspace.has_chunks():
            console.print(
                "Retrieval eval failed: no chunks found. "
                "Run ragent ingest <path> first.",
                markup=False,
                soft_wrap=True,
            )
            return 1

        if retrieval in {"semantic", "hybrid"} and not workspace.has_vector_index():
            console.print(
                "Retrieval eval failed: vector index not found. "
                "Run `ragent index build` first.",
                markup=False,
                soft_wrap=True,
            )
            return 1
        built_search = _build_search_service_for_retrieval(
            workspace,
            _as_retrieval_mode(retrieval),
            limit,
        )

        report = eval_service.evaluate(
            cases=cases,
            search_service=built_search.search_service,
            limit=limit,
            retrieval_mode=cast(RetrievalMode, retrieval),
            retrieval_method=built_search.retrieval_method,
            cases_path=cases_path,
            workspace_path=workspace.root_path,
            embedding_provider=built_search.embedding_provider,
            embedding_model=built_search.embedding_model,
            index_path=built_search.index_path,
            fusion_method=built_search.fusion_method,
            rrf_k=built_search.rrf_k,
            lexical_weight=built_search.lexical_weight,
            semantic_weight=built_search.semantic_weight,
        )
        report_payload = report.model_dump(mode="json")
        written_report_path = workspace.write_retrieval_eval_report(
            report_payload,
            report_path,
        )
        run_dir = workspace.write_retrieval_eval_run(
            report_payload,
            written_report_path,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        console.print(
            f"Retrieval eval failed: {exc}",
            markup=False,
            soft_wrap=True,
        )
        return 1

    finished_at = datetime.now(UTC)
    trace = build_retrieval_eval_trace(
        cases_path=Path(cases_path),
        retrieval_mode=retrieval,
        retrieval_method=report.retrieval_method,
        limit=report.limit,
        case_count=report.case_count,
        passed_count=report.passed_count,
        failed_count=report.failed_count,
        metrics=report.metrics,
        report_path=written_report_path,
        run_dir=run_dir,
        started_at=started_at,
        finished_at=finished_at,
        embedding_provider=built_search.embedding_provider,
        embedding_model=built_search.embedding_model,
        index_path=built_search.index_path,
        fusion_method=built_search.fusion_method,
        rrf_k=built_search.rrf_k,
        lexical_weight=built_search.lexical_weight,
        semantic_weight=built_search.semantic_weight,
    )
    trace_path = workspace.write_trace(trace)
    _print_retrieval_eval_summary(
        console,
        report,
        written_report_path,
        run_dir,
        trace_path,
    )
    return 0


def _print_retrieval_eval_summary(
    console: Console,
    report: RetrievalEvalReport,
    report_path: Path,
    run_dir: Path,
    trace_path: Path,
) -> None:
    console.print("Retrieval evaluation")
    console.print()
    console.print(f"Cases: {report.case_count}")
    console.print(f"Retrieval mode: {report.retrieval_mode}")
    console.print(f"Limit: {report.limit}")
    console.print()
    console.print(f"Passed: {report.passed_count}")
    console.print(f"Failed: {report.failed_count}")
    console.print(f"hit@1: {report.metrics['hit@1']:.4f}")
    console.print(f"hit@3: {report.metrics['hit@3']:.4f}")
    console.print(f"hit@5: {report.metrics['hit@5']:.4f}")
    console.print(f"hit@{report.limit} requested: {report.metrics['hit@k']:.4f}")
    console.print(f"MRR: {report.metrics['mrr']:.4f}")
    console.print(f"recall@{report.limit} requested: {report.metrics['recall@k']:.4f}")
    console.print(
        f"Avg retrieval latency: {report.metrics['avg_retrieval_latency_ms']:.4f} ms"
    )
    console.print(f"Avg retrieved count: {report.metrics['avg_retrieved_count']:.4f}")
    console.print(
        "Avg retrieved context chars: "
        f"{report.metrics['avg_retrieved_context_chars']:.4f}"
    )
    console.print(
        "Avg estimated context tokens: "
        f"{report.metrics['avg_estimated_context_tokens']:.4f}"
    )
    console.print()
    failed_results = [result for result in report.results if not result.passed]
    if not failed_results:
        console.print("Failed cases: none")
    else:
        console.print("Failed cases:")
    for result in failed_results:
        rank_text = result.rank if result.rank is not None else "none"
        actual_top = [
            str(top_result.get("chunk_id", ""))
            for top_result in result.top_results
        ]
        console.print(
            f"- {result.id} | rank: {rank_text} | query: {result.query}",
            soft_wrap=True,
        )
        console.print(f"  expected chunks: {result.expected_chunk_ids}")
        console.print(f"  expected sources: {result.expected_source_paths}")
        console.print(f"  actual top{report.limit}: {actual_top}")
    failure_breakdown = _retrieval_eval_failure_breakdown(report)
    if failure_breakdown:
        console.print()
        console.print("Failure breakdown:")
        for failure_type, count in sorted(failure_breakdown.items()):
            console.print(f"- {failure_type}: {count}")
    console.print()
    console.print(f"Report path: {report_path}")
    console.print(f"Run directory: {run_dir}")
    console.print(f"Saved trace to: {trace_path}")


def _retrieval_eval_failure_breakdown(
    report: RetrievalEvalReport,
) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for result in report.results:
        if result.passed:
            continue
        failure_type = result.failure_type or "unknown"
        breakdown[failure_type] = breakdown.get(failure_type, 0) + 1
    return breakdown


def _print_trace(console: Console, trace: dict[str, object]) -> None:
    console.print(f"Trace ID: {trace.get('trace_id', '')}", soft_wrap=True)
    console.print(f"Operation: {trace.get('operation', '')}")
    console.print(f"Status: {trace.get('status', '')}")
    console.print(f"Started at: {trace.get('started_at', '')}")
    console.print(f"Finished at: {trace.get('finished_at', '')}")
    console.print()
    console.print("Steps:")
    steps = trace.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    for index, step in enumerate(steps, start=1):
        step_name = step.get("name", "") if isinstance(step, dict) else ""
        console.print(f"{index}. {step_name}")

    console.print()
    console.print("Metadata:")
    metadata = trace.get("metadata", {})
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            console.print(f"- {key}: {value}", soft_wrap=True)


def _print_trace_warnings(console: Console, warnings: list[str]) -> None:
    if not warnings:
        return
    console.print()
    console.print("Warnings:")
    for warning in warnings:
        console.print(f"- {warning}", soft_wrap=True)


def _handle_search(
    console: Console,
    workspace_path: str,
    query: str,
    limit: int,
    retrieval: str,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0

    started_at = datetime.now(UTC)
    try:
        if retrieval in {"semantic", "hybrid"} and not workspace.has_vector_index():
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
            workspace,
            _as_retrieval_mode(retrieval),
            limit,
        )

        results = built_search.search_service.search(query, limit)
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
        lexical_weight=built_search.lexical_weight,
        semantic_weight=built_search.semantic_weight,
        candidate_limit=built_search.candidate_limit,
        embedding_provider=built_search.embedding_provider,
        embedding_model=built_search.embedding_model,
        index_path=built_search.index_path,
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
            result.start_char,
            result.end_char,
            result.metadata,
        )
        console.print(
            f"{index}. score={result.score:g} | {result.chunk_id}",
            soft_wrap=True,
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
        generation_service = GenerationService.from_config(config)
        if retrieval in {"semantic", "hybrid"} and not workspace.has_vector_index():
            console.print(
                "Ask failed: vector index not found. "
                "Run `ragent index build` first.",
                markup=False,
                soft_wrap=True,
            )
            return 1
        built_search = _build_search_service_for_retrieval(
            workspace,
            _as_retrieval_mode(retrieval),
            limit,
            config=config,
        )
        ask_service = AskService(
            workspace,
            generation_service=generation_service,
            search_service=built_search.search_service,
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
        lexical_weight=built_search.lexical_weight,
        semantic_weight=built_search.semantic_weight,
        embedding_provider=built_search.embedding_provider,
        embedding_model=built_search.embedding_model,
        index_path=built_search.index_path,
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
    console: Console,
    index: int,
    search_result: SearchResult,
) -> None:
    range_text = _format_search_range(
        search_result.start_char,
        search_result.end_char,
        search_result.metadata,
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
    console: Console,
    index: int,
    search_result: SearchResult,
) -> None:
    range_text = _format_search_range(
        search_result.start_char,
        search_result.end_char,
        search_result.metadata,
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
