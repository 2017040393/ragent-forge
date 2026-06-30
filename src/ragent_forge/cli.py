from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime

from rich.console import Console

from ragent_forge.app.models import ContextPack, WorkspaceStatus
from ragent_forge.app.services.ask_service import AskService
from ragent_forge.app.services.chunk_service import ChunkService, make_preview
from ragent_forge.app.services.ingest_service import IngestService
from ragent_forge.app.services.search_service import LexicalSearchService
from ragent_forge.app.services.trace_service import (
    build_ask_retrieval_trace,
    build_ingest_trace,
    build_search_trace,
)
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.tui.main import RagentForgeApp


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
        help="Ingest local Markdown/TXT knowledge folders.",
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

    search_parser = subparsers.add_parser(
        "search",
        help="Search generated chunks with simple lexical matching.",
    )
    search_parser.add_argument("query", help="Lexical query to search for.")
    search_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to search.",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to show.",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Preview retrieved context for a question without answer generation.",
    )
    ask_parser.add_argument("question", help="Question to ask.")
    ask_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to search.",
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

    if args.command == "tui":
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

    if args.command == "traces":
        if args.traces_command == "latest":
            return _handle_traces_latest(console, args.workspace)
        parser.print_help()
        return 0

    if args.command == "search":
        return _handle_search(console, args.workspace, args.query, args.limit)

    if args.command == "ask":
        return _handle_ask(
            console,
            args.workspace,
            args.question,
            args.limit,
            args.show_prompt,
            args.max_context_chars,
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
        console.print(
            f"{chunk.get('chunk_id', '')} | "
            f"{chunk.get('source_path', '')} | "
            f"{_format_char_range(chunk)} | "
            f"{make_preview(str(chunk.get('text', '')))}",
            soft_wrap=True,
        )

    if total_count > limit:
        console.print(
            f"Showing {len(chunks)} of {total_count} chunks. "
            "Use --limit to show more."
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
    return f"{start_char}-{end_char}"


def _print_no_chunks(console: Console) -> None:
    console.print("No chunks found. Run ragent ingest <path> first.")


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

    console.print(f"Trace ID: {trace.get('trace_id', '')}", soft_wrap=True)
    console.print(f"Operation: {trace.get('operation', '')}")
    console.print(f"Status: {trace.get('status', '')}")
    console.print(f"Started at: {trace.get('started_at', '')}")
    console.print(f"Finished at: {trace.get('finished_at', '')}")
    console.print()
    console.print("Steps:")
    for index, step in enumerate(trace.get("steps", []), start=1):
        step_name = step.get("name", "") if isinstance(step, dict) else ""
        console.print(f"{index}. {step_name}")

    console.print()
    console.print("Metadata:")
    metadata = trace.get("metadata", {})
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            console.print(f"- {key}: {value}", soft_wrap=True)
    return 0


def _handle_search(
    console: Console,
    workspace_path: str,
    query: str,
    limit: int,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0

    started_at = datetime.now(UTC)
    search_service = LexicalSearchService(workspace)
    try:
        results = search_service.search(query, limit)
        total_chunks = search_service.count_chunks()
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Search failed:[/bold red] {exc}")
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
    )
    trace_path = workspace.write_trace(trace)

    console.print(f"Search query: {query}")
    if not results:
        console.print("No matches found.")
        console.print(f"Saved trace to: {trace_path}")
        return 0

    console.print(f"Results: {len(results)}")
    console.print()
    for index, result in enumerate(results, start=1):
        range_text = _format_search_range(result.start_char, result.end_char)
        console.print(
            f"{index}. score={result.score:g} | {result.chunk_id}",
            soft_wrap=True,
        )
        console.print(f"Source: {result.source_path}", soft_wrap=True)
        console.print(f"Range: {range_text}")
        console.print(f"Preview: {make_preview(result.text)}")
        console.print()
    console.print(f"Saved trace to: {trace_path}")
    return 0


def _format_search_range(start_char: int | None, end_char: int | None) -> str:
    return f"{start_char}-{end_char}"


def _handle_ask(
    console: Console,
    workspace_path: str,
    question: str,
    limit: int,
    show_prompt: bool,
    max_context_chars: int,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0

    started_at = datetime.now(UTC)
    ask_service = AskService(workspace)
    try:
        result = ask_service.retrieve_context(question, limit, max_context_chars)
        total_chunks = ask_service.count_chunks()
    except (OSError, ValueError) as exc:
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
        context_chunk_count=len(result.context_pack.context_chunks),
        total_context_chars=result.context_pack.total_context_chars,
        prompt_preview_shown=show_prompt,
        max_context_chars=max_context_chars,
        started_at=started_at,
        finished_at=finished_at,
    )
    trace_path = workspace.write_trace(trace)

    console.print("Ask pipeline: retrieval-only mode")
    console.print()
    console.print(f"Question: {question}")
    console.print("Generation: not implemented yet.")
    console.print()

    if not result.results:
        console.print("No retrieved context found.")
        console.print()
        if show_prompt:
            _print_context_pack(console, result.context_pack)
        console.print(f"Saved trace to: {trace_path}")
        return 0

    console.print("Retrieved context:")
    for index, search_result in enumerate(result.results, start=1):
        range_text = _format_search_range(
            search_result.start_char,
            search_result.end_char,
        )
        console.print(
            f"{index}. score={search_result.score:g} | {search_result.chunk_id}",
            soft_wrap=True,
        )
        console.print(f"Source: {search_result.source_path}", soft_wrap=True)
        console.print(f"Range: {range_text}")
        console.print(f"Preview: {make_preview(search_result.text)}")
        console.print()
    if show_prompt:
        _print_context_pack(console, result.context_pack)
    console.print(f"Saved trace to: {trace_path}")
    return 0


def _print_context_pack(console: Console, context_pack: ContextPack) -> None:
    console.print("Context pack:")
    console.print(f"Context chunks: {len(context_pack.context_chunks)}")
    console.print(f"Total context chars: {context_pack.total_context_chars}")
    console.print(f"Retrieval method: {context_pack.retrieval_method}")
    console.print()
    console.print("Prompt preview:")
    console.print(context_pack.prompt_preview, soft_wrap=True)
    console.print()
