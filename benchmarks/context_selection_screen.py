from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from benchmarks.retrieval_baseline import (
    collect_git_state,
    collect_runtime_environment,
    sha256_file,
)
from ragent_forge.app.services.evaluation.baseline import (
    BaselineFileSpec,
    BaselineGitState,
    BaselineResolvedFile,
    BaselineRuntimeEnvironment,
    GitCommit,
    result_fingerprint,
)
from ragent_forge.app.services.evaluation.contracts import (
    RetrievalEvalCaseResult,
)
from ragent_forge.app.services.evaluation.metrics import round_metric
from ragent_forge.app.services.evaluation.screening import (
    RetrievalScreenReport,
    ScreenCaseGroup,
    ScreenGroupRole,
    ScreenRunArtifact,
)
from ragent_forge.core.retrieval.context_selection import (
    ContextSelectionPolicy,
    select_ranked_prefix_with_token_budget,
)
from ragent_forge.infrastructure.storage import atomic_write_text

DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "context_selection_screen_manifest_e4a.json"
)

ContextScreenMode = Literal["semantic", "hybrid"]
ContextSelectionGateName = Literal[
    "semantic_top5_hits_retained",
    "hybrid_top5_hits_retained",
    "hybrid_top5_context_tokens",
    "all_contexts_nonempty",
    "all_contexts_within_budget",
    "ranked_prefix_preserved",
]


class ContextParentRunSpec(BaselineFileSpec):
    retrieval_mode: ContextScreenMode
    limit: Literal[5] = 5
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContextParentScreenSpec(BaselineFileSpec):
    variant_id: str = Field(min_length=1)
    evaluation_git_commit: GitCommit
    workspace_snapshot_id: str = Field(min_length=1)
    hybrid_top5_gated_avg_tokens: float = Field(gt=0)
    runs: list[ContextParentRunSpec] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _fixed_top5_runs(self) -> Self:
        modes = [run.retrieval_mode for run in self.runs]
        if set(modes) != {"semantic", "hybrid"} or len(modes) != len(set(modes)):
            raise ValueError("context screen requires semantic@5 and hybrid@5 runs")
        self.runs = sorted(
            self.runs,
            key=lambda run: 0 if run.retrieval_mode == "semantic" else 1,
        )
        return self


class ContextSelectionVariantSpec(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    role: Literal["candidate"] = "candidate"
    description: str = Field(min_length=1)
    selection_policy: ContextSelectionPolicy
    candidate_limit: Literal[5] = 5
    max_context_tokens: int = Field(gt=0)
    characters_per_token: int = Field(gt=0)

    @model_validator(mode="after")
    def _requires_ranked_prefix_budget(self) -> Self:
        if self.selection_policy != "ranked_prefix_token_budget_v1":
            raise ValueError("E4 context screen requires ranked-prefix token budget")
        return self

    @property
    def max_context_chars(self) -> int:
        return self.max_context_tokens * self.characters_per_token


class ContextSelectionPromotionSpec(BaseModel):
    max_semantic_top5_hit_losses: Literal[0] = 0
    max_hybrid_top5_hit_losses: Literal[0] = 0
    max_hybrid_top5_token_ratio: float = Field(gt=0)
    min_selected_chunks_per_case: int = Field(gt=0)


class ContextSelectionScreenManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parent_screen: ContextParentScreenSpec
    variant: ContextSelectionVariantSpec
    case_groups: list[ScreenCaseGroup] = Field(min_length=5, max_length=5)
    promotion: ContextSelectionPromotionSpec

    @model_validator(mode="after")
    def _fixed_case_groups(self) -> Self:
        expected_roles: set[ScreenGroupRole] = {
            "stable_control",
            "semantic_opportunity",
            "wrong_section_challenge",
            "hard_miss",
            "boundary_canary",
        }
        roles = [group.role for group in self.case_groups]
        if set(roles) != expected_roles or len(roles) != len(set(roles)):
            raise ValueError("context screen requires exactly one group per role")
        case_ids = self.selected_case_ids
        if len(case_ids) != 16 or len(case_ids) != len(set(case_ids)):
            raise ValueError("context screen requires 16 globally unique cases")
        return self

    @property
    def selected_case_ids(self) -> list[str]:
        return [case_id for group in self.case_groups for case_id in group.case_ids]

    @property
    def gated_case_ids(self) -> list[str]:
        return [
            case_id
            for group in self.case_groups
            if group.role != "boundary_canary"
            for case_id in group.case_ids
        ]


class RankedContextItem(BaseModel):
    rank: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    text_chars: int = Field(ge=0)


class ContextSelectionCaseResult(BaseModel):
    case_id: str
    group: str
    role: ScreenGroupRole
    parent_hit: bool
    relevant_ranks: list[int]
    ranked_chunk_ids: list[str]
    selected_chunk_ids: list[str]
    selected_ranks: list[int]
    selected_count: int = Field(ge=0)
    selected_context_chars: int = Field(ge=0)
    estimated_context_tokens: int = Field(ge=0)
    selected_hit: bool
    hit_retained: bool
    context_nonempty: bool
    budget_respected: bool
    ranked_prefix_preserved: bool
    mapping_coverage: float = Field(ge=0, le=1)


class ContextSelectionMetrics(BaseModel):
    case_count: int = Field(gt=0)
    parent_hits: int = Field(ge=0)
    selected_hits: int = Field(ge=0)
    retained_hits: int = Field(ge=0)
    lost_hit_case_ids: list[str]
    average_selected_chunks: float = Field(ge=0)
    average_selected_context_chars: float = Field(ge=0)
    average_estimated_context_tokens: float = Field(ge=0)
    maximum_estimated_context_tokens: int = Field(ge=0)


class ResolvedContextParentRun(BaseModel):
    retrieval_mode: ContextScreenMode
    limit: Literal[5] = 5
    file: BaselineResolvedFile
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContextSelectionRunReport(BaseModel):
    schema_version: Literal[1] = 1
    variant_id: str
    retrieval_mode: ContextScreenMode
    limit: Literal[5] = 5
    parent_run: ResolvedContextParentRun
    selection_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: ContextSelectionMetrics
    cases: list[ContextSelectionCaseResult] = Field(min_length=16, max_length=16)


class ContextSelectionGateResult(BaseModel):
    name: ContextSelectionGateName
    passed: bool
    observed: str
    requirement: str
    case_ids: list[str] = Field(default_factory=list)


class ResolvedContextParentScreen(BaseModel):
    summary: BaselineResolvedFile
    variant_id: str
    evaluation_git_commit: GitCommit
    workspace_snapshot_id: str
    hybrid_top5_gated_avg_tokens: float = Field(gt=0)
    runs: list[ResolvedContextParentRun] = Field(min_length=2, max_length=2)


class ContextSelectionScreenReport(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    description: str
    measured_at: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git: BaselineGitState
    runtime: BaselineRuntimeEnvironment
    parent_screen: ResolvedContextParentScreen
    variant: ContextSelectionVariantSpec
    selected_case_ids: list[str] = Field(min_length=16, max_length=16)
    configurations: list[ContextSelectionRunReport] = Field(
        min_length=2,
        max_length=2,
    )
    gates: list[ContextSelectionGateResult] = Field(min_length=6, max_length=6)
    hybrid_top5_gated_avg_tokens: float = Field(ge=0)
    hybrid_top5_token_ratio: float = Field(ge=0)
    valid: bool
    promoted: bool


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> ContextSelectionScreenManifest:
    return ContextSelectionScreenManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def run_context_selection_screen(
    manifest: ContextSelectionScreenManifest,
    *,
    manifest_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
    git_state: BaselineGitState,
    runtime_environment: BaselineRuntimeEnvironment,
) -> ContextSelectionScreenReport:
    root = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    source_manifest_path = Path(manifest_path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Context screen output directory already exists: {destination}"
        )

    parent_summary, resolved_parent = _load_parent_screen(manifest, root)
    role_by_case_id: dict[str, tuple[str, ScreenGroupRole]] = {}
    for group in manifest.case_groups:
        for case_id in group.case_ids:
            role_by_case_id[case_id] = (group.name, group.role)
    configurations: list[ContextSelectionRunReport] = []
    destination.mkdir(parents=True)
    (destination / "runs").mkdir()
    atomic_write_text(
        destination / "manifest.json",
        manifest.model_dump_json(indent=2) + "\n",
    )
    for run_spec, resolved_run in zip(
        manifest.parent_screen.runs,
        resolved_parent.runs,
        strict=True,
    ):
        artifact = _load_parent_run(
            manifest,
            parent_summary,
            run_spec,
            resolved_run,
            root,
        )
        configuration = build_context_selection_run(
            manifest,
            artifact=artifact,
            parent_run=resolved_run,
            role_by_case_id=role_by_case_id,
        )
        configurations.append(configuration)
        atomic_write_text(
            destination / "runs" / f"{run_spec.retrieval_mode}-k5.json",
            configuration.model_dump_json(indent=2) + "\n",
        )

    hybrid_configuration = next(
        item for item in configurations if item.retrieval_mode == "hybrid"
    )
    hybrid_gated_cases = [
        case
        for case in hybrid_configuration.cases
        if case.case_id in set(manifest.gated_case_ids)
    ]
    hybrid_gated_avg_tokens = round_metric(
        sum(case.estimated_context_tokens for case in hybrid_gated_cases)
        / len(hybrid_gated_cases)
    )
    hybrid_token_ratio = round_metric(
        hybrid_gated_avg_tokens
        / manifest.parent_screen.hybrid_top5_gated_avg_tokens
    )
    gates = evaluate_context_selection_gates(
        manifest,
        configurations,
        hybrid_top5_token_ratio=hybrid_token_ratio,
    )
    valid = not git_state.dirty
    report = ContextSelectionScreenReport(
        benchmark=manifest.name,
        description=manifest.description,
        measured_at=datetime.now(UTC).isoformat(),
        manifest_path=_display_path(source_manifest_path, root),
        manifest_sha256=sha256_file(source_manifest_path, "text_lf"),
        git=git_state,
        runtime=runtime_environment,
        parent_screen=resolved_parent,
        variant=manifest.variant,
        selected_case_ids=manifest.selected_case_ids,
        configurations=configurations,
        gates=gates,
        hybrid_top5_gated_avg_tokens=hybrid_gated_avg_tokens,
        hybrid_top5_token_ratio=hybrid_token_ratio,
        valid=valid,
        promoted=valid and all(gate.passed for gate in gates),
    )
    atomic_write_text(
        destination / "summary.json",
        report.model_dump_json(indent=2) + "\n",
    )
    return report


def build_context_selection_run(
    manifest: ContextSelectionScreenManifest,
    *,
    artifact: ScreenRunArtifact,
    parent_run: ResolvedContextParentRun,
    role_by_case_id: Mapping[str, tuple[str, ScreenGroupRole]],
) -> ContextSelectionRunReport:
    cases = [
        _select_case(
            result,
            group=role_by_case_id[result.id][0],
            role=role_by_case_id[result.id][1],
            variant=manifest.variant,
        )
        for result in artifact.evaluation.results
    ]
    lost = [case.case_id for case in cases if not case.hit_retained]
    metrics = ContextSelectionMetrics(
        case_count=len(cases),
        parent_hits=sum(case.parent_hit for case in cases),
        selected_hits=sum(case.selected_hit for case in cases),
        retained_hits=sum(case.parent_hit and case.selected_hit for case in cases),
        lost_hit_case_ids=lost,
        average_selected_chunks=round_metric(
            sum(case.selected_count for case in cases) / len(cases)
        ),
        average_selected_context_chars=round_metric(
            sum(case.selected_context_chars for case in cases) / len(cases)
        ),
        average_estimated_context_tokens=round_metric(
            sum(case.estimated_context_tokens for case in cases) / len(cases)
        ),
        maximum_estimated_context_tokens=max(
            case.estimated_context_tokens for case in cases
        ),
    )
    return ContextSelectionRunReport(
        variant_id=manifest.variant.id,
        retrieval_mode=parent_run.retrieval_mode,
        parent_run=parent_run,
        selection_fingerprint_sha256=_selection_fingerprint(cases),
        metrics=metrics,
        cases=cases,
    )


def evaluate_context_selection_gates(
    manifest: ContextSelectionScreenManifest,
    configurations: Sequence[ContextSelectionRunReport],
    *,
    hybrid_top5_token_ratio: float,
) -> list[ContextSelectionGateResult]:
    by_mode = {
        configuration.retrieval_mode: configuration
        for configuration in configurations
    }
    if set(by_mode) != {"semantic", "hybrid"}:
        raise ValueError("context screen requires semantic and hybrid configurations")
    semantic_losses = by_mode["semantic"].metrics.lost_hit_case_ids
    hybrid_losses = by_mode["hybrid"].metrics.lost_hit_case_ids
    promotion = manifest.promotion
    empty = [
        f"{configuration.retrieval_mode}:{case.case_id}"
        for configuration in configurations
        for case in configuration.cases
        if case.selected_count < promotion.min_selected_chunks_per_case
    ]
    over_budget = [
        f"{configuration.retrieval_mode}:{case.case_id}"
        for configuration in configurations
        for case in configuration.cases
        if not case.budget_respected
    ]
    invalid_prefix = [
        f"{configuration.retrieval_mode}:{case.case_id}"
        for configuration in configurations
        for case in configuration.cases
        if not case.ranked_prefix_preserved
    ]
    return [
        ContextSelectionGateResult(
            name="semantic_top5_hits_retained",
            passed=(len(semantic_losses) <= promotion.max_semantic_top5_hit_losses),
            observed=f"losses={len(semantic_losses)}",
            requirement=f"losses<={promotion.max_semantic_top5_hit_losses}",
            case_ids=semantic_losses,
        ),
        ContextSelectionGateResult(
            name="hybrid_top5_hits_retained",
            passed=(len(hybrid_losses) <= promotion.max_hybrid_top5_hit_losses),
            observed=f"losses={len(hybrid_losses)}",
            requirement=f"losses<={promotion.max_hybrid_top5_hit_losses}",
            case_ids=hybrid_losses,
        ),
        ContextSelectionGateResult(
            name="hybrid_top5_context_tokens",
            passed=(hybrid_top5_token_ratio <= promotion.max_hybrid_top5_token_ratio),
            observed=f"ratio={hybrid_top5_token_ratio:.4f}",
            requirement=f"ratio<={promotion.max_hybrid_top5_token_ratio:.4f}",
        ),
        ContextSelectionGateResult(
            name="all_contexts_nonempty",
            passed=(len(empty) == 0),
            observed=f"below_minimum={len(empty)}",
            requirement=(
                "below_minimum="
                f"0 (min={promotion.min_selected_chunks_per_case})"
            ),
            case_ids=empty,
        ),
        ContextSelectionGateResult(
            name="all_contexts_within_budget",
            passed=(len(over_budget) == 0),
            observed=f"over_budget={len(over_budget)}",
            requirement="over_budget=0",
            case_ids=over_budget,
        ),
        ContextSelectionGateResult(
            name="ranked_prefix_preserved",
            passed=(len(invalid_prefix) == 0),
            observed=f"invalid_prefix={len(invalid_prefix)}",
            requirement="invalid_prefix=0",
            case_ids=invalid_prefix,
        ),
    ]


def _select_case(
    result: RetrievalEvalCaseResult,
    *,
    group: str,
    role: ScreenGroupRole,
    variant: ContextSelectionVariantSpec,
) -> ContextSelectionCaseResult:
    ranked = [RankedContextItem.model_validate(item) for item in result.top_results]
    expected_ranks = list(range(1, len(ranked) + 1))
    if [item.rank for item in ranked] != expected_ranks:
        raise ValueError(f"Parent ranks are not contiguous for case {result.id}")
    if [item.chunk_id for item in ranked] != result.actual_chunk_ids:
        raise ValueError(f"Parent ranked chunk ids mismatch for case {result.id}")
    selected = select_ranked_prefix_with_token_budget(
        ranked,
        limit=variant.candidate_limit,
        max_context_tokens=variant.max_context_tokens,
        characters_per_token=variant.characters_per_token,
        text_length=lambda item: item.text_chars,
    )
    relevant = set(result.relevant_result_ranks)
    selected_ranks = [item.rank for item in selected]
    parent_hit = bool(relevant.intersection(expected_ranks))
    selected_hit = bool(relevant.intersection(selected_ranks))
    selected_chars = sum(item.text_chars for item in selected)
    estimated_tokens = math.ceil(selected_chars / variant.characters_per_token)
    selected_ids = [item.chunk_id for item in selected]
    ranked_ids = [item.chunk_id for item in ranked]
    mapping_coverage = result.mapping_coverage
    if mapping_coverage is None:
        raise ValueError(f"Parent mapping coverage is missing for case {result.id}")
    return ContextSelectionCaseResult(
        case_id=result.id,
        group=group,
        role=role,
        parent_hit=parent_hit,
        relevant_ranks=result.relevant_result_ranks,
        ranked_chunk_ids=ranked_ids,
        selected_chunk_ids=selected_ids,
        selected_ranks=selected_ranks,
        selected_count=len(selected),
        selected_context_chars=selected_chars,
        estimated_context_tokens=estimated_tokens,
        selected_hit=selected_hit,
        hit_retained=(not parent_hit or selected_hit),
        context_nonempty=(len(selected) >= 1),
        budget_respected=(estimated_tokens <= variant.max_context_tokens),
        ranked_prefix_preserved=(selected_ids == ranked_ids[: len(selected_ids)]),
        mapping_coverage=mapping_coverage,
    )


def _load_parent_screen(
    manifest: ContextSelectionScreenManifest,
    repository_root: Path,
) -> tuple[RetrievalScreenReport, ResolvedContextParentScreen]:
    spec = manifest.parent_screen
    summary_path = _resolve_repository_path(repository_root, spec.path)
    summary_digest = sha256_file(summary_path, spec.hash_mode)
    if summary_digest != spec.sha256:
        raise ValueError("Parent context screen summary hash mismatch")
    summary = RetrievalScreenReport.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    if summary.variant.id != spec.variant_id:
        raise ValueError("Parent context screen variant mismatch")
    if not summary.valid:
        raise ValueError("Parent context screen is structurally invalid")
    if summary.git.commit != spec.evaluation_git_commit:
        raise ValueError("Parent context screen evaluation commit mismatch")
    if summary.workspace.snapshot_id != spec.workspace_snapshot_id:
        raise ValueError("Parent context screen workspace snapshot mismatch")
    if summary.selected_case_ids != manifest.selected_case_ids:
        raise ValueError("Parent context screen case order mismatch")
    hybrid = next(
        configuration
        for configuration in summary.configurations
        if configuration.retrieval_mode == "hybrid" and configuration.limit == 5
    )
    baseline_tokens = hybrid.baseline_gated_metrics.avg_selected_context_tokens.average
    if baseline_tokens != spec.hybrid_top5_gated_avg_tokens:
        raise ValueError("Parent Hybrid@5 gated token reference mismatch")
    resolved_runs = [
        ResolvedContextParentRun(
            retrieval_mode=run.retrieval_mode,
            file=_resolve_file(run, repository_root),
            result_fingerprint_sha256=run.result_fingerprint_sha256,
        )
        for run in spec.runs
    ]
    return summary, ResolvedContextParentScreen(
        summary=BaselineResolvedFile(
            path=_display_path(summary_path, repository_root),
            sha256=summary_digest,
            hash_mode=spec.hash_mode,
        ),
        variant_id=spec.variant_id,
        evaluation_git_commit=spec.evaluation_git_commit,
        workspace_snapshot_id=spec.workspace_snapshot_id,
        hybrid_top5_gated_avg_tokens=baseline_tokens,
        runs=resolved_runs,
    )


def _load_parent_run(
    manifest: ContextSelectionScreenManifest,
    parent_summary: RetrievalScreenReport,
    spec: ContextParentRunSpec,
    resolved: ResolvedContextParentRun,
    repository_root: Path,
) -> ScreenRunArtifact:
    artifact_path = _resolve_repository_path(repository_root, spec.path)
    artifact = ScreenRunArtifact.model_validate_json(
        artifact_path.read_text(encoding="utf-8")
    )
    parent = manifest.parent_screen
    if artifact.variant_id != parent.variant_id:
        raise ValueError("Parent context run variant mismatch")
    if artifact.git_commit != parent.evaluation_git_commit:
        raise ValueError("Parent context run evaluation commit mismatch")
    if artifact.workspace_snapshot_id != parent.workspace_snapshot_id:
        raise ValueError("Parent context run workspace snapshot mismatch")
    evaluation = artifact.evaluation
    if evaluation.retrieval_mode != spec.retrieval_mode or evaluation.limit != 5:
        raise ValueError("Parent context run configuration mismatch")
    if [result.id for result in evaluation.results] != manifest.selected_case_ids:
        raise ValueError("Parent context run case order mismatch")
    if any(result.mapping_coverage != 1.0 for result in evaluation.results):
        raise ValueError("Parent context run mapping coverage is incomplete")
    fingerprint = result_fingerprint(evaluation.results)
    if fingerprint != spec.result_fingerprint_sha256:
        raise ValueError("Parent context run result fingerprint mismatch")
    summary_configuration = next(
        configuration
        for configuration in parent_summary.configurations
        if configuration.retrieval_mode == spec.retrieval_mode
        and configuration.limit == 5
    )
    if summary_configuration.result_fingerprint_sha256 != fingerprint:
        raise ValueError("Parent summary and run fingerprint mismatch")
    if resolved.file.path != _display_path(artifact_path, repository_root):
        raise ValueError("Resolved parent run path mismatch")
    return artifact


def _resolve_file(
    spec: BaselineFileSpec,
    repository_root: Path,
) -> BaselineResolvedFile:
    path = _resolve_repository_path(repository_root, spec.path)
    digest = sha256_file(path, spec.hash_mode)
    if digest != spec.sha256:
        raise ValueError(f"Context screen input hash mismatch: {spec.path}")
    return BaselineResolvedFile(
        path=_display_path(path, repository_root),
        sha256=digest,
        hash_mode=spec.hash_mode,
    )


def _selection_fingerprint(cases: Sequence[ContextSelectionCaseResult]) -> str:
    payload = [
        {
            "case_id": case.case_id,
            "selected_chunk_ids": case.selected_chunk_ids,
            "selected_context_chars": case.selected_context_chars,
            "estimated_context_tokens": case.estimated_context_tokens,
        }
        for case in cases
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to(repository_root):
        raise ValueError(f"Context screen path escapes repository root: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"Context screen input file not found: {path}")
    return path


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _print_summary(report: ContextSelectionScreenReport, output_dir: Path) -> None:
    print(f"Context selection screen: {report.benchmark}")
    print(f"Variant: {report.variant.id}")
    for configuration in report.configurations:
        metrics = configuration.metrics
        print(
            f"{configuration.retrieval_mode}@5: "
            f"hits={metrics.selected_hits}/{metrics.parent_hits} "
            f"avg-tokens={metrics.average_estimated_context_tokens:.4f}"
        )
    print(
        "Hybrid@5 gated token ratio: "
        f"{report.hybrid_top5_token_ratio:.4f}"
    )
    print("Gates:")
    for gate in report.gates:
        print(
            f"  {'PASS' if gate.passed else 'FAIL'} {gate.name}: "
            f"{gate.observed} ({gate.requirement})"
        )
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Valid: {report.valid}")
    print(f"Promoted: {report.promoted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a frozen retrieval ranking through context selection."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to a checked-in context selection screen manifest.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for the context screen artifacts.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty-tree diagnostic run that cannot be promoted.",
    )
    args = parser.parse_args(argv)

    try:
        repository_root = Path.cwd().resolve()
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        repository_root = Path(completed.stdout.strip()).resolve()
        git_state = collect_git_state(repository_root)
        if git_state.dirty and not args.allow_dirty:
            raise ValueError(
                "Context screening requires a clean Git tree; use --allow-dirty "
                "only for a non-promotable diagnostic run"
            )
        report = run_context_selection_screen(
            load_manifest(args.manifest),
            manifest_path=args.manifest,
            repository_root=repository_root,
            output_dir=args.output_dir,
            git_state=git_state,
            runtime_environment=collect_runtime_environment(),
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Context selection screen failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, Path(args.output_dir).resolve())
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
