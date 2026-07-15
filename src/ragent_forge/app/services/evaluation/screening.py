from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from ragent_forge.app.services.evaluation.baseline import (
    BaselineCacheState,
    BaselineCorpusSpec,
    BaselineCutoffMetricDistribution,
    BaselineCutoffMetrics,
    BaselineDatasetSpec,
    BaselineEmbeddingSpec,
    BaselineFileSpec,
    BaselineGitState,
    BaselineIngestSpec,
    BaselineResolvedDataset,
    BaselineResolvedFile,
    BaselineRuntimeEnvironment,
    BaselineWorkspaceState,
    GitCommit,
    aggregate_cutoff_metrics,
    compute_cutoff_metrics,
    result_fingerprint,
)
from ragent_forge.app.services.evaluation.contracts import (
    FailureType,
    RetrievalEvalCaseResult,
    RetrievalEvalReport,
)
from ragent_forge.core.retrieval.representations import (
    EmbeddingRepresentation,
    QueryEmbeddingRepresentation,
)

ScreenMode = Literal["semantic", "hybrid"]
ScreenLimit = Literal[5, 20]
ScreenGroupRole = Literal[
    "stable_control",
    "semantic_opportunity",
    "wrong_section_challenge",
    "hard_miss",
    "boundary_canary",
]
ScreenVariantRole = Literal["baseline", "candidate"]
ScreenTransition = Literal[
    "retained",
    "gained",
    "lost",
    "unchanged_miss",
    "boundary",
]
ScreenGateName = Literal[
    "semantic_stable_hits_retained",
    "no_new_missed_source",
    "semantic_challenge_gain",
    "hybrid_top5_net_nonnegative",
    "hybrid_top5_context_tokens",
    "complete_evidence_mapping",
]


class ScreenParentBaselineSpec(BaselineFileSpec):
    required_repetitions: Literal[3] = 3


class ScreenVariantSpec(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    role: ScreenVariantRole
    description: str = Field(min_length=1)
    document_embedding_representation: EmbeddingRepresentation
    query_embedding_representation: QueryEmbeddingRepresentation
    workspace_build_git_commit: GitCommit
    expected_chunk_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_index_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScreenCaseGroup(BaseModel):
    name: str = Field(min_length=1)
    role: ScreenGroupRole
    case_ids: list[str] = Field(min_length=1)

    @field_validator("case_ids")
    @classmethod
    def _unique_non_empty_case_ids(cls, case_ids: list[str]) -> list[str]:
        normalized = [case_id.strip() for case_id in case_ids]
        if any(not case_id for case_id in normalized):
            raise ValueError("screen case ids must be non-empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("screen case ids must be unique within a group")
        return normalized


class ScreenWorkloadSpec(BaseModel):
    retrieval_modes: list[ScreenMode] = Field(min_length=2, max_length=2)
    limits: list[ScreenLimit] = Field(min_length=2, max_length=2)
    repetitions: Literal[1] = 1

    @model_validator(mode="after")
    def _fixed_diagnostic_matrix(self) -> Self:
        if set(self.retrieval_modes) != {"semantic", "hybrid"}:
            raise ValueError("screening requires semantic and hybrid modes")
        if set(self.limits) != {5, 20}:
            raise ValueError("screening requires independent limits 5 and 20")
        self.retrieval_modes = ["semantic", "hybrid"]
        self.limits = [5, 20]
        return self


class ScreenPromotionSpec(BaseModel):
    max_stable_semantic_losses: int = Field(ge=0)
    max_new_missed_source_results: int = Field(ge=0)
    min_new_semantic_top5_hits: int = Field(ge=0)
    min_new_semantic_top20_hits: int = Field(ge=0)
    min_hybrid_top5_hit_delta: int
    max_hybrid_top5_token_ratio: float = Field(gt=0)


class RetrievalScreenManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parent_baseline: ScreenParentBaselineSpec
    dataset: BaselineDatasetSpec
    corpus: BaselineCorpusSpec
    ingest: BaselineIngestSpec
    embedding: BaselineEmbeddingSpec
    variant: ScreenVariantSpec
    workload: ScreenWorkloadSpec
    case_groups: list[ScreenCaseGroup] = Field(min_length=5, max_length=5)
    promotion: ScreenPromotionSpec

    @model_validator(mode="after")
    def _fixed_unique_case_groups(self) -> Self:
        expected_roles: set[ScreenGroupRole] = {
            "stable_control",
            "semantic_opportunity",
            "wrong_section_challenge",
            "hard_miss",
            "boundary_canary",
        }
        roles = [group.role for group in self.case_groups]
        if set(roles) != expected_roles or len(roles) != len(set(roles)):
            raise ValueError("screening requires exactly one group for every role")
        case_ids = [case_id for group in self.case_groups for case_id in group.case_ids]
        if len(case_ids) != 16:
            raise ValueError("screening requires exactly 16 selected cases")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("screen case ids must be globally unique")
        return self

    @property
    def selected_case_ids(self) -> list[str]:
        return [
            case_id
            for group in self.case_groups
            for case_id in group.case_ids
        ]

    @property
    def gated_case_ids(self) -> list[str]:
        return [
            case_id
            for group in self.case_groups
            if group.role != "boundary_canary"
            for case_id in group.case_ids
        ]


class ScreenWorkspaceFingerprints(BaseModel):
    chunk_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScreenResolvedParentBaseline(BaseModel):
    summary: BaselineResolvedFile
    git_commit: GitCommit
    workspace_snapshot_id: str
    required_repetitions: Literal[3] = 3


class ScreenQueryCacheSummary(BaseModel):
    artifact_path: str
    source_path: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str
    model: str
    query_representation: QueryEmbeddingRepresentation
    embedding_dim: int = Field(gt=0)
    entry_count: int = Field(gt=0)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)


class ScreenRunArtifact(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    variant_id: str
    git_commit: GitCommit
    workspace_snapshot_id: str
    cache: BaselineCacheState
    cache_reuse_valid: bool
    query_cache_hits: int = Field(ge=0)
    query_cache_misses: int = Field(ge=0)
    evaluation: RetrievalEvalReport


class ScreenBaselineCaseOutcome(BaseModel):
    hit_count: int = Field(ge=0, le=3)
    ranks: list[int | None] = Field(min_length=3, max_length=3)
    failure_types: list[FailureType | None] = Field(min_length=3, max_length=3)


class ScreenCaseComparison(BaseModel):
    case_id: str
    group: str
    role: ScreenGroupRole
    baseline: ScreenBaselineCaseOutcome
    candidate_hit: bool
    candidate_rank: int | None = None
    candidate_failure_type: FailureType | None = None
    candidate_mapping_coverage: float | None = Field(default=None, ge=0, le=1)
    transition: ScreenTransition


class ScreenConfigurationReport(BaseModel):
    retrieval_mode: ScreenMode
    limit: ScreenLimit
    artifact_path: str
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_metrics: BaselineCutoffMetricDistribution
    candidate_metrics: BaselineCutoffMetrics
    baseline_gated_metrics: BaselineCutoffMetricDistribution
    candidate_gated_metrics: BaselineCutoffMetrics
    cache_reuse_valid: bool
    query_cache_hits: int = Field(ge=0)
    query_cache_misses: int = Field(ge=0)
    cases: list[ScreenCaseComparison] = Field(min_length=16, max_length=16)


class ScreenGateResult(BaseModel):
    name: ScreenGateName
    passed: bool
    observed: str
    requirement: str
    case_ids: list[str] = Field(default_factory=list)


class RetrievalScreenReport(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    description: str
    measured_at: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git: BaselineGitState
    runtime: BaselineRuntimeEnvironment
    parent_baseline: ScreenResolvedParentBaseline
    variant: ScreenVariantSpec
    dataset: BaselineResolvedDataset
    corpus: list[BaselineResolvedFile]
    workspace: BaselineWorkspaceState
    workspace_fingerprints: ScreenWorkspaceFingerprints
    selected_case_ids: list[str] = Field(min_length=16, max_length=16)
    query_cache: ScreenQueryCacheSummary
    configurations: list[ScreenConfigurationReport] = Field(
        min_length=4,
        max_length=4,
    )
    gates: list[ScreenGateResult] = Field(min_length=6, max_length=6)
    valid: bool
    promotion_applicable: bool
    promoted: bool | None


def build_screen_configuration(
    manifest: RetrievalScreenManifest,
    *,
    mode: ScreenMode,
    limit: ScreenLimit,
    artifact_path: str,
    baseline_reports: Sequence[RetrievalEvalReport],
    candidate_report: RetrievalEvalReport,
    cache_reuse_valid: bool,
    query_cache_hits: int,
    query_cache_misses: int,
) -> ScreenConfigurationReport:
    if len(baseline_reports) != manifest.parent_baseline.required_repetitions:
        raise ValueError("screening requires three parent baseline trials")
    if any(
        report.retrieval_mode != mode or report.limit != limit
        for report in baseline_reports
    ):
        raise ValueError("parent baseline configuration does not match screen")
    if (
        candidate_report.retrieval_mode != mode
        or candidate_report.limit != limit
    ):
        raise ValueError("candidate configuration does not match screen")

    selected = manifest.selected_case_ids
    candidate_by_id = _results_by_id(candidate_report.results)
    baseline_by_trial = [
        _results_by_id(report.results) for report in baseline_reports
    ]
    _require_selected_cases(selected, candidate_by_id, "candidate")
    for index, baseline_by_id in enumerate(baseline_by_trial, start=1):
        _require_selected_cases(selected, baseline_by_id, f"baseline trial {index}")

    group_by_case = {
        case_id: group
        for group in manifest.case_groups
        for case_id in group.case_ids
    }
    comparisons = []
    for case_id in selected:
        group = group_by_case[case_id]
        baseline_results = [trial[case_id] for trial in baseline_by_trial]
        candidate_result = candidate_by_id[case_id]
        baseline = ScreenBaselineCaseOutcome(
            hit_count=sum(result.passed for result in baseline_results),
            ranks=[result.rank for result in baseline_results],
            failure_types=[result.failure_type for result in baseline_results],
        )
        comparisons.append(
            ScreenCaseComparison(
                case_id=case_id,
                group=group.name,
                role=group.role,
                baseline=baseline,
                candidate_hit=candidate_result.passed,
                candidate_rank=candidate_result.rank,
                candidate_failure_type=candidate_result.failure_type,
                candidate_mapping_coverage=candidate_result.mapping_coverage,
                transition=_transition(
                    group.role,
                    baseline.hit_count,
                    candidate_result.passed,
                ),
            )
        )

    baseline_metrics = [
        compute_cutoff_metrics(
            [trial[case_id] for case_id in selected],
            limit,
        )
        for trial in baseline_by_trial
    ]
    gated = manifest.gated_case_ids
    baseline_gated_metrics = [
        compute_cutoff_metrics(
            [trial[case_id] for case_id in gated],
            limit,
        )
        for trial in baseline_by_trial
    ]
    candidate_results = [candidate_by_id[case_id] for case_id in selected]
    candidate_gated_results = [candidate_by_id[case_id] for case_id in gated]
    return ScreenConfigurationReport(
        retrieval_mode=mode,
        limit=limit,
        artifact_path=artifact_path,
        result_fingerprint_sha256=result_fingerprint(candidate_results),
        baseline_metrics=aggregate_cutoff_metrics(baseline_metrics),
        candidate_metrics=compute_cutoff_metrics(candidate_results, limit),
        baseline_gated_metrics=aggregate_cutoff_metrics(
            baseline_gated_metrics
        ),
        candidate_gated_metrics=compute_cutoff_metrics(
            candidate_gated_results,
            limit,
        ),
        cache_reuse_valid=cache_reuse_valid,
        query_cache_hits=query_cache_hits,
        query_cache_misses=query_cache_misses,
        cases=comparisons,
    )


def evaluate_screen_gates(
    manifest: RetrievalScreenManifest,
    configurations: Sequence[ScreenConfigurationReport],
) -> list[ScreenGateResult]:
    by_configuration = {
        (configuration.retrieval_mode, configuration.limit): configuration
        for configuration in configurations
    }
    expected_keys: set[tuple[ScreenMode, ScreenLimit]] = {
        ("semantic", 5),
        ("semantic", 20),
        ("hybrid", 5),
        ("hybrid", 20),
    }
    if set(by_configuration) != expected_keys:
        raise ValueError("screen gate evaluation requires the complete matrix")

    semantic_top5 = by_configuration[("semantic", 5)]
    semantic_top20 = by_configuration[("semantic", 20)]
    hybrid_top5 = by_configuration[("hybrid", 5)]
    gated = set(manifest.gated_case_ids)
    stable_roles: set[ScreenGroupRole] = {
        "stable_control",
        "semantic_opportunity",
    }
    challenge_roles: set[ScreenGroupRole] = {
        "wrong_section_challenge",
        "hard_miss",
    }

    stable_losses = [
        case.case_id
        for case in semantic_top5.cases
        if case.role in stable_roles and not case.candidate_hit
    ]
    stable_parent_mismatches = [
        case.case_id
        for case in semantic_top5.cases
        if case.role in stable_roles and case.baseline.hit_count != 3
    ]
    if stable_parent_mismatches:
        raise ValueError(
            "stable screen cases do not match the parent baseline: "
            f"{stable_parent_mismatches}"
        )

    new_missed_source = sorted(
        f"{configuration.retrieval_mode}@{configuration.limit}:{case.case_id}"
        for configuration in configurations
        for case in configuration.cases
        if case.case_id in gated
        and case.candidate_failure_type == "missed_source"
        and "missed_source" not in case.baseline.failure_types
    )
    top5_gains = [
        case.case_id
        for case in semantic_top5.cases
        if case.role in challenge_roles
        and case.baseline.hit_count == 0
        and case.candidate_hit
    ]
    top20_gains = [
        case.case_id
        for case in semantic_top20.cases
        if case.role in challenge_roles
        and case.baseline.hit_count == 0
        and case.candidate_hit
    ]
    challenge_gain_passed = (
        len(top5_gains) >= manifest.promotion.min_new_semantic_top5_hits
        or len(top20_gains) >= manifest.promotion.min_new_semantic_top20_hits
    )

    hybrid_gated_cases = [
        case for case in hybrid_top5.cases if case.case_id in gated
    ]
    baseline_hybrid_hits = sum(
        case.baseline.hit_count >= 2 for case in hybrid_gated_cases
    )
    candidate_hybrid_hits = sum(
        case.candidate_hit for case in hybrid_gated_cases
    )
    hybrid_hit_delta = candidate_hybrid_hits - baseline_hybrid_hits

    baseline_tokens = (
        hybrid_top5.baseline_gated_metrics.avg_selected_context_tokens.average
    )
    candidate_tokens = (
        hybrid_top5.candidate_gated_metrics.avg_selected_context_tokens
    )
    token_ratio = (
        candidate_tokens / baseline_tokens
        if baseline_tokens
        else float("inf")
    )
    incomplete_mapping = sorted(
        f"{configuration.retrieval_mode}@{configuration.limit}:{case.case_id}"
        for configuration in configurations
        for case in configuration.cases
        if case.candidate_mapping_coverage != 1.0
    )

    return [
        ScreenGateResult(
            name="semantic_stable_hits_retained",
            passed=(
                len(stable_losses)
                <= manifest.promotion.max_stable_semantic_losses
            ),
            observed=f"losses={len(stable_losses)}",
            requirement=(
                "losses<="
                f"{manifest.promotion.max_stable_semantic_losses}"
            ),
            case_ids=stable_losses,
        ),
        ScreenGateResult(
            name="no_new_missed_source",
            passed=(
                len(new_missed_source)
                <= manifest.promotion.max_new_missed_source_results
            ),
            observed=f"new_results={len(new_missed_source)}",
            requirement=(
                "new_results<="
                f"{manifest.promotion.max_new_missed_source_results}"
            ),
            case_ids=new_missed_source,
        ),
        ScreenGateResult(
            name="semantic_challenge_gain",
            passed=challenge_gain_passed,
            observed=(
                f"new_top5={len(top5_gains)},new_top20={len(top20_gains)}"
            ),
            requirement=(
                f"new_top5>={manifest.promotion.min_new_semantic_top5_hits} "
                "or "
                f"new_top20>={manifest.promotion.min_new_semantic_top20_hits}"
            ),
            case_ids=sorted(set(top5_gains + top20_gains)),
        ),
        ScreenGateResult(
            name="hybrid_top5_net_nonnegative",
            passed=(
                hybrid_hit_delta
                >= manifest.promotion.min_hybrid_top5_hit_delta
            ),
            observed=f"hit_delta={hybrid_hit_delta}",
            requirement=(
                "hit_delta>="
                f"{manifest.promotion.min_hybrid_top5_hit_delta}"
            ),
        ),
        ScreenGateResult(
            name="hybrid_top5_context_tokens",
            passed=(
                token_ratio
                <= manifest.promotion.max_hybrid_top5_token_ratio
            ),
            observed=f"ratio={token_ratio:.4f}",
            requirement=(
                "ratio<="
                f"{manifest.promotion.max_hybrid_top5_token_ratio:.4f}"
            ),
        ),
        ScreenGateResult(
            name="complete_evidence_mapping",
            passed=not incomplete_mapping,
            observed=f"incomplete_results={len(incomplete_mapping)}",
            requirement="incomplete_results=0",
            case_ids=incomplete_mapping,
        ),
    ]


def _results_by_id(
    results: Sequence[RetrievalEvalCaseResult],
) -> dict[str, RetrievalEvalCaseResult]:
    result_by_id = {result.id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError("retrieval report contains duplicate case ids")
    return result_by_id


def _require_selected_cases(
    selected: Sequence[str],
    results: Mapping[str, RetrievalEvalCaseResult],
    label: str,
) -> None:
    missing = sorted(set(selected) - set(results))
    if missing:
        raise ValueError(f"{label} is missing selected cases: {missing}")


def _transition(
    role: ScreenGroupRole,
    baseline_hit_count: int,
    candidate_hit: bool,
) -> ScreenTransition:
    if role == "boundary_canary" or baseline_hit_count not in {0, 3}:
        return "boundary"
    if baseline_hit_count == 3:
        return "retained" if candidate_hit else "lost"
    return "gained" if candidate_hit else "unchanged_miss"
