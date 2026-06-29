from __future__ import annotations

import argparse
from collections.abc import Sequence

from rich.console import Console

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
        console.print(
            "[bold]Ingestion stub:[/bold] local Markdown/TXT ingestion for "
            f"[cyan]{args.path}[/cyan] will be implemented in v0.1."
        )
        return 0

    if args.command == "ask":
        console.print(
            "[bold]Ask stub:[/bold] the inspectable RAG pipeline is not implemented "
            f"yet. Received question: [cyan]{args.question}[/cyan]"
        )
        return 0

    parser.print_help()
    return 0
