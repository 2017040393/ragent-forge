from __future__ import annotations

import argparse

from ragent_forge.core.retrieval.representations import EMBEDDING_REPRESENTATIONS
from ragent_forge.core.retrieval.types import RETRIEVAL_MODES

RETRIEVAL_CHOICES = list(RETRIEVAL_MODES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragent",
        description=(
            "A local-first TUI workbench for inspectable Agentic RAG workflows "
            "over personal knowledge bases."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    tui_parser = subparsers.add_parser("tui", help="Launch the local Textual TUI.")
    tui_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest local Markdown/TXT/PDF knowledge folders or files."
    )
    ingest_parser.add_argument("path", help="Path to a local knowledge folder or file.")
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Maximum characters per chunk."
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
        "status", help="Show local RAGentForge workspace status."
    )
    status_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    workspace_parser = subparsers.add_parser(
        "workspace", help="Inspect or migrate workspace storage layout."
    )
    workspace_subparsers = workspace_parser.add_subparsers(dest="workspace_command")
    workspace_migrate_parser = workspace_subparsers.add_parser(
        "migrate", help="Migrate a legacy flat workspace to generation storage."
    )
    workspace_migrate_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to migrate.",
    )
    workspace_migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the migration plan without writing files.",
    )
    config_parser = subparsers.add_parser(
        "config", help="Inspect or initialize local RAGentForge configuration."
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_show_parser = config_subparsers.add_parser(
        "show", help="Show the effective local configuration."
    )
    config_show_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    config_init_parser = config_subparsers.add_parser(
        "init", help="Write the default local configuration."
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
        "chunks", help="Inspect generated chunks in a local workspace."
    )
    chunks_subparsers = chunks_parser.add_subparsers(dest="chunks_command")
    chunks_list_parser = chunks_subparsers.add_parser(
        "list", help="List generated chunks."
    )
    chunks_list_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    chunks_list_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of chunks to show."
    )
    chunks_show_parser = chunks_subparsers.add_parser(
        "show", help="Show a generated chunk by exact chunk id."
    )
    chunks_show_parser.add_argument("chunk_id", help="Exact chunk id to inspect.")
    chunks_show_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    traces_parser = subparsers.add_parser(
        "traces", help="Inspect local operation traces."
    )
    traces_subparsers = traces_parser.add_subparsers(dest="traces_command")
    traces_latest_parser = traces_subparsers.add_parser(
        "latest", help="Show the latest local operation trace."
    )
    traces_latest_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    traces_list_parser = traces_subparsers.add_parser(
        "list", help="List local operation trace history."
    )
    traces_list_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    traces_list_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of traces to show."
    )
    traces_show_parser = traces_subparsers.add_parser(
        "show", help="Show a local operation trace by exact trace id."
    )
    traces_show_parser.add_argument("trace_id", help="Exact trace id to inspect.")
    traces_show_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    eval_parser = subparsers.add_parser("eval", help="Run local evaluation workflows.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    retrieval_eval_parser = eval_subparsers.add_parser(
        "retrieval",
        help=(
            "Evaluate lexical, BM25, semantic, or hybrid retrieval quality "
            "with JSONL cases."
        ),
    )
    retrieval_eval_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to evaluate.",
    )
    retrieval_eval_parser.add_argument(
        "--cases", required=True, help="Path to a JSONL retrieval eval cases file."
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
    retrieval_compare_parser = eval_subparsers.add_parser(
        "compare", help="Compare retrieval modes and top-k limits against JSONL cases."
    )
    retrieval_compare_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to evaluate.",
    )
    retrieval_compare_parser.add_argument(
        "--cases", required=True, help="Path to a JSONL retrieval eval cases file."
    )
    retrieval_compare_parser.add_argument(
        "--retrieval",
        default="lexical,bm25,semantic,hybrid",
        help="Comma-separated retrieval modes to compare.",
    )
    retrieval_compare_parser.add_argument(
        "--limit",
        default="1,3,5",
        help="Comma-separated positive top-k limits to compare.",
    )
    retrieval_compare_parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the JSON retrieval compare report.",
    )
    retrieval_compare_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed mode/limit run.",
    )
    eval_generate_parser = eval_subparsers.add_parser(
        "generate", help="Generate span-based synthetic retrieval eval JSONL cases."
    )
    eval_generate_parser.add_argument(
        "--source", required=True, help="Path to a source document file or directory."
    )
    eval_generate_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory for configuration.",
    )
    eval_generate_parser.add_argument(
        "--output", required=True, help="Path to write generated JSONL eval cases."
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
        "index", help="Build or inspect the local semantic vector index."
    )
    index_subparsers = index_parser.add_subparsers(dest="index_command")
    index_build_parser = index_subparsers.add_parser(
        "build", help="Build the local semantic vector index from generated chunks."
    )
    index_build_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to index.",
    )
    index_build_parser.add_argument(
        "--embedding-representation",
        choices=EMBEDDING_REPRESENTATIONS,
        default="raw_chunk_text_v1",
        help="Document text representation sent to the embedding provider.",
    )
    index_status_parser = index_subparsers.add_parser(
        "status", help="Show local semantic vector index status."
    )
    index_status_parser.add_argument(
        "--workspace",
        default=".ragent",
        help="Local RAGentForge workspace directory to inspect.",
    )
    search_parser = subparsers.add_parser(
        "search",
        help=(
            "Search generated chunks with lexical, BM25, semantic, or hybrid retrieval."
        ),
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
        "--limit", type=int, default=10, help="Maximum number of results to show."
    )
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question using local retrieval and optional generation."
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
        "--limit", type=int, default=5, help="Maximum number of context chunks to show."
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
