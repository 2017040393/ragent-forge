from __future__ import annotations

import argparse
from collections.abc import Sequence

from rich.console import Console

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
        help="Stub for ingesting local Markdown/TXT knowledge folders.",
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

    if args.command == "ask":
        console.print(
            "[bold]Ask stub:[/bold] the inspectable RAG pipeline is not implemented "
            f"yet. Received question: [cyan]{args.question}[/cyan]"
        )
        return 0

    parser.print_help()
    return 0
