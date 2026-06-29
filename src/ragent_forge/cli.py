from __future__ import annotations

import argparse
from collections.abc import Sequence

from rich.console import Console

from ragent_forge.app.models import WorkspaceStatus
from ragent_forge.app.services.ingest_service import IngestService
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

    ask_parser = subparsers.add_parser(
        "ask",
        help="Stub for asking a question against the local knowledge base.",
    )
    ask_parser.add_argument("question", help="Question to ask.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.command == "tui":
        RagentForgeApp().run()
        return 0

    if args.command == "ingest":
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

        console.print("[bold green]Ingest complete[/bold green]")
        console.print(f"Source: [cyan]{result.source_path}[/cyan]")
        console.print(f"Documents: {result.document_count}")
        console.print(f"Chunks: {result.chunk_count}")
        console.print(f"Skipped files: {result.skipped_count}")
        console.print(f"Chunk size: {result.metadata['chunk_size']}")
        console.print(f"Chunk overlap: {result.metadata['chunk_overlap']}")
        console.print(f"Saved chunks to: {chunks_path}")
        console.print(f"Saved summary to: {summary_path}")
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

    if args.command == "ask":
        console.print(
            "[bold]Ask stub:[/bold] the inspectable RAG pipeline is not implemented "
            f"yet. Received question: [cyan]{args.question}[/cyan]"
        )
        return 0

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
    chunk_count = workspace_status.chunk_count_from_file or summary.get(
        "chunk_count",
        0,
    )
    console.print(f"Chunks: {chunk_count}")
    console.print(f"Skipped files: {summary.get('skipped_count', 0)}")
    console.print()
    console.print(f"Chunks file: {workspace_status.chunks_path}")
    console.print(f"Summary file: {workspace_status.latest_summary_path}")
