from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from rich.console import Console

from ragent_forge.app.models import AppConfig
from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.eval_dataset_generation_service import (
    EvalDatasetGenerationReport,
    EvalDatasetGenerationService,
    TextGenerationClient,
)
from ragent_forge.app.services.evaluation.runner import (
    RetrievalEvalReport,
    RetrievalEvalService,
)
from ragent_forge.app.services.evidence_span_service import EvidenceSpanService
from ragent_forge.app.services.retrieval_compare_service import (
    RetrievalCompareReport,
    RetrievalCompareRun,
)
from ragent_forge.app.services.trace_service import build_retrieval_eval_trace
from ragent_forge.cli.handlers.retrieval import (
    _as_retrieval_mode,
    _build_search_service_for_retrieval,
)
from ragent_forge.composition import build_text_generation_client
from ragent_forge.core.retrieval.types import RETRIEVAL_MODES, RetrievalMode
from ragent_forge.infrastructure.eval_output import (
    write_generated_eval_jsonl as write_jsonl,
)
from ragent_forge.infrastructure.local_workspace import LocalWorkspace

RETRIEVAL_CHOICES = list(RETRIEVAL_MODES)


def _parse_retrieval_modes(value: str) -> list[RetrievalMode]:
    modes: list[RetrievalMode] = []
    for raw_mode in value.split(","):
        mode = raw_mode.strip()
        if not mode:
            raise ValueError("--retrieval must not contain empty values")
        if mode not in RETRIEVAL_CHOICES:
            choices = ", ".join(RETRIEVAL_CHOICES)
            raise ValueError(
                f"--retrieval contains invalid mode '{mode}'. "
                f"Expected one of: {choices}"
            )
        retrieval_mode = cast(RetrievalMode, mode)
        if retrieval_mode not in modes:
            modes.append(retrieval_mode)
    if not modes:
        raise ValueError("--retrieval must include at least one mode")
    return modes


def _parse_positive_int_list(value: str, *, option_name: str) -> list[int]:
    integers: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            raise ValueError(f"{option_name} must not contain empty values")
        try:
            parsed = int(item)
        except ValueError as exc:
            raise ValueError(
                f"{option_name} contains non-integer value '{item}'"
            ) from exc
        if parsed < 1:
            raise ValueError(f"{option_name} values must be positive integers")
        if parsed not in integers:
            integers.append(parsed)
    if not integers:
        raise ValueError(f"{option_name} must include at least one value")
    return integers


def _build_text_generation_client(config: AppConfig) -> TextGenerationClient:
    if config.generation.provider == "openai_responses":
        return build_text_generation_client(config)
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
    *,
    text_generation_client_builder: Callable[
        [AppConfig], TextGenerationClient
    ] = _build_text_generation_client,
) -> int:
    try:
        if questions_per_span < 1:
            raise ValueError("questions_per_span must be greater than 0")
        if max_cases is not None and max_cases < 0:
            raise ValueError("max_cases must be greater than or equal to 0")
        output = Path(output_path)
        if output.exists() and (not overwrite) and (not dry_run):
            raise FileExistsError(f"Output JSONL already exists: {output}")
        workspace = LocalWorkspace(workspace_path)
        config = ConfigService(workspace).load()
        if config.generation.provider == "null" and (not dry_run):
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
        text_generation_client = text_generation_client_builder(config)
        report = EvalDatasetGenerationService(
            generator=text_generation_client, questions_per_span=questions_per_span
        ).generate(spans, max_cases=max_cases)
        if report.generated_count == 0:
            _print_eval_generate_empty_failure(console, report)
            return 1
        written_output_path = write_jsonl(report.cases, output, overwrite=overwrite)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        console.print(f"Eval generation failed: {exc}", markup=False, soft_wrap=True)
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
    console: Console, report: EvalDatasetGenerationReport
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
        f"Next: ragent eval retrieval --cases {output_path} "
        f"--workspace {workspace_path} --retrieval lexical --limit 5",
        soft_wrap=True,
    )


def _handle_eval_compare(
    console: Console,
    workspace_path: str,
    cases_path: str,
    retrieval: str,
    limit: str,
    output_path: str | None,
    fail_fast: bool,
) -> int:
    workspace = LocalWorkspace(workspace_path)
    eval_service = RetrievalEvalService()
    try:
        retrieval_modes = _parse_retrieval_modes(retrieval)
        limits = _parse_positive_int_list(limit, option_name="--limit")
        cases = eval_service.load_cases(cases_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        console.print(f"Retrieval compare failed: {exc}", markup=False, soft_wrap=True)
        return 1
    if not workspace.has_chunks():
        console.print(
            "Retrieval compare failed: no chunks found. "
            "Run ragent ingest <path> first.",
            markup=False,
            soft_wrap=True,
        )
        return 1
    runs: list[RetrievalCompareRun] = []
    stop_matrix = False
    for retrieval_mode in retrieval_modes:
        if stop_matrix:
            break
        for top_k in limits:
            missing_vector_index = retrieval_mode in {"semantic", "hybrid"} and (
                not workspace.has_vector_index()
            )
            if missing_vector_index:
                runs.append(
                    RetrievalCompareRun(
                        retrieval_mode=retrieval_mode,
                        limit=top_k,
                        status="failed",
                        case_count=len(cases),
                        error="vector index not found; run `ragent index build` first",
                    )
                )
                if fail_fast:
                    stop_matrix = True
                    break
                continue
            try:
                built_search = _build_search_service_for_retrieval(
                    workspace, retrieval_mode, top_k
                )
                report = eval_service.evaluate(
                    cases=cases,
                    search_service=built_search.retrieval_engine,
                    limit=top_k,
                    retrieval_mode=retrieval_mode,
                    retrieval_method=built_search.retrieval_method,
                    cases_path=cases_path,
                    workspace_path=workspace.root_path,
                    embedding_provider=built_search.embedding_provider,
                    embedding_model=built_search.embedding_model,
                    index_path=built_search.index_path,
                    fusion_method=built_search.fusion_method,
                    rrf_k=built_search.rrf_k,
                    sparse_method=built_search.sparse_method,
                    dense_method=built_search.dense_method,
                    sparse_weight=built_search.sparse_weight,
                    dense_weight=built_search.dense_weight,
                    lexical_weight=built_search.lexical_weight,
                    semantic_weight=built_search.semantic_weight,
                    workspace=workspace,
                )
                report_payload = report.model_dump(mode="json")
                written_report_path = workspace.write_retrieval_eval_report(
                    report_payload
                )
                run_dir = workspace.write_retrieval_eval_run(
                    report_payload, written_report_path
                )
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                runs.append(
                    RetrievalCompareRun(
                        retrieval_mode=retrieval_mode,
                        limit=top_k,
                        status="failed",
                        case_count=len(cases),
                        error=str(exc),
                    )
                )
                if fail_fast:
                    stop_matrix = True
                    break
                continue
            runs.append(
                RetrievalCompareRun(
                    retrieval_mode=report.retrieval_mode,
                    retrieval_method=report.retrieval_method,
                    limit=report.limit,
                    status="success",
                    metrics=report.metrics,
                    passed_count=report.passed_count,
                    failed_count=report.failed_count,
                    case_count=report.case_count,
                    report_path=str(written_report_path),
                    run_dir=str(run_dir),
                    failure_breakdown=_retrieval_eval_failure_breakdown(report),
                )
            )
    compare_report = _build_retrieval_compare_report(
        cases_path=cases_path,
        workspace=workspace,
        retrieval_modes=retrieval_modes,
        limits=limits,
        runs=runs,
    )
    try:
        compare_report_path = workspace.write_retrieval_compare_report(
            compare_report.model_dump(mode="json"), output_path
        )
    except OSError as exc:
        console.print(f"Retrieval compare failed: {exc}", markup=False, soft_wrap=True)
        return 1
    _print_retrieval_compare_summary(
        console, compare_report, compare_report_path, workspace.eval_runs_dir
    )
    if compare_report.success_count == 0:
        return 1
    if fail_fast and compare_report.failed_count > 0:
        return 1
    return 0


def _build_retrieval_compare_report(
    *,
    cases_path: str,
    workspace: LocalWorkspace,
    retrieval_modes: list[RetrievalMode],
    limits: list[int],
    runs: list[RetrievalCompareRun],
) -> RetrievalCompareReport:
    success_count = sum(1 for run in runs if run.status == "success")
    failed_count = len(runs) - success_count
    return RetrievalCompareReport(
        cases_path=str(Path(cases_path)),
        workspace=str(workspace.root_path),
        retrieval_modes=list(retrieval_modes),
        limits=limits,
        run_count=len(runs),
        success_count=success_count,
        failed_count=failed_count,
        runs=runs,
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
        if retrieval in {"semantic", "hybrid"} and (not workspace.has_vector_index()):
            console.print(
                "Retrieval eval failed: vector index not found. "
                "Run `ragent index build` first.",
                markup=False,
                soft_wrap=True,
            )
            return 1
        built_search = _build_search_service_for_retrieval(
            workspace, _as_retrieval_mode(retrieval), limit
        )
        report = eval_service.evaluate(
            cases=cases,
            search_service=built_search.retrieval_engine,
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
            sparse_method=built_search.sparse_method,
            dense_method=built_search.dense_method,
            sparse_weight=built_search.sparse_weight,
            dense_weight=built_search.dense_weight,
            lexical_weight=built_search.lexical_weight,
            semantic_weight=built_search.semantic_weight,
            workspace=workspace,
        )
        report_payload = report.model_dump(mode="json")
        written_report_path = workspace.write_retrieval_eval_report(
            report_payload, report_path
        )
        run_dir = workspace.write_retrieval_eval_run(
            report_payload, written_report_path
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"Retrieval eval failed: {exc}", markup=False, soft_wrap=True)
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
        sparse_method=built_search.sparse_method,
        dense_method=built_search.dense_method,
        sparse_weight=built_search.sparse_weight,
        dense_weight=built_search.dense_weight,
        lexical_weight=built_search.lexical_weight,
        semantic_weight=built_search.semantic_weight,
        retrieval_stages=report.retrieval_pipeline,
        stage_latency_ms=report.stage_latency_ms,
    )
    trace_path = workspace.write_trace(trace)
    _print_retrieval_eval_summary(
        console, report, written_report_path, run_dir, trace_path
    )
    return 0


def _print_retrieval_compare_summary(
    console: Console,
    report: RetrievalCompareReport,
    compare_report_path: Path,
    eval_runs_dir: Path,
) -> None:
    console.print("Retrieval comparison")
    console.print(f"Cases: {_compare_case_count(report)}")
    console.print(f"Modes: {', '.join(report.retrieval_modes)}")
    console.print(f"Limits: {', '.join(str(limit) for limit in report.limits)}")
    console.print()
    console.print(
        "mode      k   status   hit@k  rec@k  pre@k  nDCG   MRR    p95ms      fail"
    )
    for run in report.runs:
        failures = (
            str(run.failed_count if run.failed_count is not None else 0)
            if run.status == "success"
            else str(run.error or "")
        )
        console.print(
            f"{run.retrieval_mode:<9} {run.limit:<3} {run.status:<8} "
            f"{_compare_metric_text(run, 'hit@k'):<6} "
            f"{_compare_metric_text(run, 'recall@k'):<6} "
            f"{_compare_metric_text(run, 'precision@k'):<6} "
            f"{_compare_metric_text(run, 'ndcg@k'):<6} "
            f"{_compare_metric_text(run, 'mrr'):<6} "
            f"{_compare_metric_text(run, 'retrieval_latency_p95_ms'):<10} "
            f"{failures}",
            soft_wrap=True,
        )
    console.print()
    console.print(f"Compare report path: {compare_report_path}")
    if report.success_count > 0:
        console.print(f"Individual run dirs: {eval_runs_dir}")


def _compare_case_count(report: RetrievalCompareReport) -> int:
    for run in report.runs:
        if run.case_count is not None:
            return run.case_count
    return 0


def _compare_metric_text(run: RetrievalCompareRun, metric_key: str) -> str:
    if run.status != "success":
        return "-"
    metric = run.metrics.get(metric_key)
    if metric is None:
        return "-"
    return f"{metric:.4f}"


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
        f"precision@{report.limit} requested: {report.metrics['precision@k']:.4f}"
    )
    console.print(f"nDCG@{report.limit}: {report.metrics['ndcg@k']:.4f}")
    console.print(
        f"Evidence coverage@{report.limit}: {report.metrics['evidence_coverage@k']:.4f}"
    )
    console.print(f"Mapping coverage: {report.metrics['mapping_coverage']:.4f}")
    console.print(
        f"Context evidence density: {report.metrics['context_evidence_density']:.4f}"
    )
    console.print(
        f"Duplicate context ratio: {report.metrics['duplicate_context_ratio']:.4f}"
    )
    console.print(
        f"Avg retrieval latency: {report.metrics['avg_retrieval_latency_ms']:.4f} ms"
    )
    console.print(
        f"Retrieval latency p50: {report.metrics['retrieval_latency_p50_ms']:.4f} ms"
    )
    console.print(
        f"Retrieval latency p95: {report.metrics['retrieval_latency_p95_ms']:.4f} ms"
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
            str(top_result.get("chunk_id", "")) for top_result in result.top_results
        ]
        console.print(
            f"- {result.id} | rank: {rank_text} | query: {result.query}", soft_wrap=True
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


def _retrieval_eval_failure_breakdown(report: RetrievalEvalReport) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for result in report.results:
        if result.passed:
            continue
        failure_type = result.failure_type or "unknown"
        breakdown[failure_type] = breakdown.get(failure_type, 0) + 1
    return breakdown
