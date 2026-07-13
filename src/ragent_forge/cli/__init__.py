from __future__ import annotations

# ruff: noqa: F401 -- this module preserves the historical CLI import surface.
from collections.abc import Sequence

from rich.console import Console

from ragent_forge.cli.handlers import (
    chunks as _chunks,
)
from ragent_forge.cli.handlers import (
    config as _config,
)
from ragent_forge.cli.handlers import (
    evaluation as _evaluation,
)
from ragent_forge.cli.handlers import (
    index as _index,
)
from ragent_forge.cli.handlers import (
    retrieval as _retrieval,
)
from ragent_forge.cli.handlers import (
    traces as _traces,
)
from ragent_forge.cli.handlers import (
    workspace as _workspace,
)
from ragent_forge.cli.handlers.chunks import (
    _format_char_range,
    _handle_chunks_list,
    _handle_chunks_show,
    _print_no_chunks,
)
from ragent_forge.cli.handlers.config import (
    _handle_config_init,
    _handle_config_show,
)
from ragent_forge.cli.handlers.evaluation import (
    _build_retrieval_compare_report,
    _build_text_generation_client,
    _compare_case_count,
    _compare_metric_text,
    _handle_eval_compare,
    _handle_eval_generate,
    _handle_eval_retrieval,
    _parse_positive_int_list,
    _parse_retrieval_modes,
    _print_eval_generate_dry_run_summary,
    _print_eval_generate_empty_failure,
    _print_eval_generate_summary,
    _print_retrieval_compare_summary,
    _print_retrieval_eval_summary,
    _retrieval_eval_failure_breakdown,
)
from ragent_forge.cli.handlers.index import (
    _handle_index_build,
    _handle_index_status,
)
from ragent_forge.cli.handlers.retrieval import (
    BuiltSearchService,
    _as_retrieval_mode,
    _build_search_service_for_retrieval,
    _format_search_range,
    _handle_ask,
    _handle_search,
    _print_context_pack,
    _print_retrieved_context,
    _print_source_line,
)
from ragent_forge.cli.handlers.traces import (
    _handle_traces_latest,
    _handle_traces_list,
    _handle_traces_show,
    _print_trace,
    _print_trace_warnings,
)
from ragent_forge.cli.handlers.workspace import _print_workspace_status
from ragent_forge.cli.parser import RETRIEVAL_CHOICES, build_parser
from ragent_forge.tui.main import RagentForgeApp


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.command is None:
        RagentForgeApp().run()
        return 0
    if args.command == "tui":
        RagentForgeApp(args.workspace).run()
        return 0
    if args.command == "workspace":
        if args.workspace_command != "migrate":
            parser.print_help()
            return 1
        return _workspace.handle_workspace_migrate(
            console,
            args.workspace,
            dry_run=args.dry_run,
        )
    if args.command == "ingest":
        return _workspace.handle_ingest(
            console,
            args.path,
            args.workspace,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    if args.command == "status":
        return _workspace.handle_status(console, args.workspace)
    if args.command == "chunks":
        if args.chunks_command == "list":
            return _chunks._handle_chunks_list(console, args.workspace, args.limit)
        if args.chunks_command == "show":
            return _chunks._handle_chunks_show(
                console,
                args.workspace,
                args.chunk_id,
            )
        parser.print_help()
        return 0
    if args.command == "config":
        if args.config_command == "show":
            return _config._handle_config_show(console, args.workspace)
        if args.config_command == "init":
            return _config._handle_config_init(
                console,
                args.workspace,
                args.overwrite,
            )
        parser.print_help()
        return 0
    if args.command == "traces":
        if args.traces_command == "latest":
            return _traces._handle_traces_latest(console, args.workspace)
        if args.traces_command == "list":
            return _traces._handle_traces_list(
                console,
                args.workspace,
                args.limit,
            )
        if args.traces_command == "show":
            return _traces._handle_traces_show(
                console,
                args.workspace,
                args.trace_id,
            )
        parser.print_help()
        return 0
    if args.command == "eval":
        if args.eval_command == "retrieval":
            return _evaluation._handle_eval_retrieval(
                console,
                args.workspace,
                args.cases,
                args.retrieval,
                args.limit,
                args.report_path,
            )
        if args.eval_command == "compare":
            return _evaluation._handle_eval_compare(
                console,
                args.workspace,
                args.cases,
                args.retrieval,
                args.limit,
                args.output,
                args.fail_fast,
            )
        if args.eval_command == "generate":
            return _evaluation._handle_eval_generate(
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
                text_generation_client_builder=_build_text_generation_client,
            )
        parser.print_help()
        return 0
    if args.command == "index":
        if args.index_command == "build":
            return _index._handle_index_build(console, args.workspace)
        if args.index_command == "status":
            return _index._handle_index_status(console, args.workspace)
        parser.print_help()
        return 0
    if args.command == "search":
        return _retrieval._handle_search(
            console,
            args.workspace,
            args.query,
            args.limit,
            args.retrieval,
        )
    if args.command == "ask":
        return _retrieval._handle_ask(
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


__all__ = [
    "BuiltSearchService",
    "RETRIEVAL_CHOICES",
    "build_parser",
    "main",
]
