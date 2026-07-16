from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from benchmarks.direction_confirmation import (
    DirectionConfirmationManifest,
    DirectionConfirmationReport,
)
from benchmarks.retrieval_baseline import (
    ValidatedBaselineInputs,
    collect_git_state,
    collect_runtime_environment,
    sha256_file,
    validate_inputs,
)
from benchmarks.retrieval_screen import (
    chunk_content_fingerprint,
    index_input_fingerprint,
)
from ragent_forge.app.services.evaluation.baseline import (
    BaselineCorpusSpec,
    BaselineDatasetSpec,
    BaselineEmbeddingSpec,
    BaselineFileSpec,
    BaselineGitState,
    BaselineIngestSpec,
    BaselineResolvedDataset,
    BaselineResolvedFile,
    BaselineRuntimeEnvironment,
    BaselineWorkloadSpec,
    BaselineWorkspaceState,
    GitCommit,
    RetrievalBaselineManifest,
    result_fingerprint,
)
from ragent_forge.app.services.evaluation.contracts import (
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
)
from ragent_forge.app.services.evaluation.metrics import round_metric
from ragent_forge.app.services.evaluation.screening import (
    ScreenRunArtifact,
    ScreenWorkspaceFingerprints,
)
from ragent_forge.app.services.vector_index_service import VectorIndexService
from ragent_forge.core.retrieval.context_fragments import (
    ContextFragment,
    FragmentScoreComponents,
    RankedFragmentCandidate,
    build_evidence_window_scorer,
    build_query_window_scorer,
    fragments_are_traceable,
    normalized_token_ngrams,
    render_fragments,
    select_ranked_fragments,
)
from ragent_forge.core.retrieval.context_selection import ContextSelectionPolicy
from ragent_forge.infrastructure.local_workspace import LocalWorkspace
from ragent_forge.infrastructure.storage import atomic_write_text

DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "fragment_packing_development_manifest_e4b.json"
)

FragmentMode = Literal["semantic", "hybrid"]
FragmentGateName = Literal[
    "oracle_evidence_reachable",
    "semantic_fragment_hits_retained",
    "hybrid_fragment_hits_retained",
    "all_candidates_represented",
    "all_fragments_traceable",
    "all_contexts_within_budget",
    "complete_evidence_mapping",
    "hybrid_context_tokens",
    "selector_gold_isolation",
]

FROZEN_SELECTOR_INPUT_FIELDS = [
    "query",
    "rank",
    "chunk_id",
    "source_path",
    "source_label",
    "page_label",
    "text",
    "metadata_signals",
]


class FragmentParentRunSpec(BaselineFileSpec):
    retrieval_mode: FragmentMode
    limit: Literal[5] = 5
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FragmentParentSpec(BaselineFileSpec):
    direction_manifest: BaselineFileSpec
    evaluation_git_commit: GitCommit
    workspace_snapshot_id: str = Field(min_length=1)
    runs: list[FragmentParentRunSpec] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _fixed_parent_runs(self) -> Self:
        modes = [run.retrieval_mode for run in self.runs]
        if set(modes) != {"semantic", "hybrid"} or len(modes) != len(set(modes)):
            raise ValueError("E4b requires unique Semantic@5 and Hybrid@5 parent runs")
        self.runs = sorted(
            self.runs,
            key=lambda run: 0 if run.retrieval_mode == "semantic" else 1,
        )
        return self


class FragmentWorkspaceSpec(BaseModel):
    build_git_commit: GitCommit
    expected_chunk_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_index_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FragmentVariantSpec(BaseModel):
    id: Literal["E4b1-ranked-query-fragments"] = "E4b1-ranked-query-fragments"
    description: str = Field(min_length=1)
    selection_policy: ContextSelectionPolicy
    candidate_limit: Literal[5] = 5
    max_context_tokens: Literal[768] = 768
    characters_per_token: Literal[4] = 4
    max_fragment_chars: Literal[640] = 640
    stride_chars: Literal[112] = 112
    evidence_ngram_size: Literal[3] = 3
    selector_input_fields: list[str] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def _fixed_fragment_contract(self) -> Self:
        if self.selection_policy != "ranked_query_fragment_budget_v1":
            raise ValueError("E4b requires ranked query fragment selection")
        if self.selector_input_fields != FROZEN_SELECTOR_INPUT_FIELDS:
            raise ValueError("E4b selector input allowlist drifted")
        return self

    @property
    def max_context_chars(self) -> int:
        return self.max_context_tokens * self.characters_per_token


class FragmentPromotionSpec(BaseModel):
    max_oracle_hit_losses: Literal[0] = 0
    max_unscorable_parent_hits: Literal[0] = 0
    max_semantic_fragment_hit_losses: Literal[0] = 0
    max_hybrid_fragment_hit_losses: Literal[0] = 0
    max_hybrid_context_token_ratio: float = Field(gt=0)
    required_candidates_per_case: Literal[5] = 5


class FragmentDevelopmentManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parent: FragmentParentSpec
    workspace: FragmentWorkspaceSpec
    variant: FragmentVariantSpec
    e0_hybrid_top5_average_context_tokens: float = Field(gt=0)
    promotion: FragmentPromotionSpec


class ResolvedFragmentParentRun(BaseModel):
    retrieval_mode: FragmentMode
    file: BaselineResolvedFile
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResolvedFragmentParent(BaseModel):
    summary: BaselineResolvedFile
    direction_manifest: BaselineResolvedFile
    evaluation_git_commit: GitCommit
    workspace_snapshot_id: str
    runs: list[ResolvedFragmentParentRun] = Field(min_length=2, max_length=2)


class FragmentScoreRecord(BaseModel):
    unique_query_tokens: int = Field(ge=0)
    query_token_occurrences: int = Field(ge=0)
    query_bigram_coverage: int = Field(ge=0)
    signal_token_coverage: int = Field(ge=0)


class FragmentRecord(BaseModel):
    rank: int = Field(ge=1, le=5)
    chunk_id: str
    source_path: str
    source_label: str
    page_label: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)
    truncated_left: bool
    truncated_right: bool
    rendered_header: str
    score: FragmentScoreRecord


class FragmentEvidenceMetrics(BaseModel):
    parent_hit: bool
    scorable: bool
    reachable_evidence_ngrams: int = Field(ge=0)
    selected_evidence_ngrams: int = Field(ge=0)
    oracle_evidence_ngrams: int = Field(ge=0)
    reachable_evidence_coverage: float | None = Field(default=None, ge=0, le=1)
    oracle_efficiency: float | None = Field(default=None, ge=0, le=1)
    oracle_retained: bool
    selector_retained: bool


class FragmentCaseResult(BaseModel):
    case_id: str
    query: str
    parent_relevant_ranks: list[int]
    parent_relevant_chunk_ids: list[str]
    parent_mapping_coverage: float
    ranked_chunk_ids: list[str] = Field(min_length=5, max_length=5)
    oracle_fragments: list[FragmentRecord] = Field(min_length=5, max_length=5)
    selector_fragments: list[FragmentRecord] = Field(min_length=5, max_length=5)
    oracle_rendered_context: str
    selector_rendered_context: str
    oracle_estimated_context_tokens: int = Field(ge=0)
    selector_estimated_context_tokens: int = Field(ge=0)
    candidates_represented: bool
    fragments_traceable: bool
    budget_respected: bool
    evidence: FragmentEvidenceMetrics


class FragmentRunMetrics(BaseModel):
    case_count: int = Field(default=50, gt=0)
    parent_hits: int = Field(ge=0)
    scorable_parent_hits: int = Field(ge=0)
    unscorable_parent_hit_case_ids: list[str]
    oracle_lost_hit_case_ids: list[str]
    selector_lost_hit_case_ids: list[str]
    average_selector_context_tokens: float = Field(ge=0)
    maximum_selector_context_tokens: int = Field(ge=0)
    average_reachable_evidence_coverage: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    minimum_reachable_evidence_coverage: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    average_oracle_efficiency: float | None = Field(default=None, ge=0, le=1)
    candidate_representation_violations: list[str]
    traceability_violations: list[str]
    budget_violations: list[str]
    mapping_violations: list[str]


class FragmentOracleMetadata(BaseModel):
    oracle_only: Literal[True] = True
    promotion_eligible: Literal[False] = False
    input_fields: list[str] = [
        "ranking",
        "chunk_text",
        "gold_evidence",
    ]


class FragmentRunArtifact(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    variant_id: str
    git_commit: GitCommit
    workspace_snapshot_id: str
    retrieval_mode: FragmentMode
    parent_ranking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selector_input_fields: list[str]
    oracle: FragmentOracleMetadata
    metrics: FragmentRunMetrics
    cases: list[FragmentCaseResult] = Field(min_length=1)


class FragmentGateResult(BaseModel):
    name: FragmentGateName
    passed: bool
    observed: str
    requirement: str
    case_ids: list[str] = Field(default_factory=list)


class FragmentDevelopmentReport(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    description: str
    measured_at: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git: BaselineGitState
    runtime: BaselineRuntimeEnvironment
    parent: ResolvedFragmentParent
    dataset: BaselineResolvedDataset
    corpus: list[BaselineResolvedFile]
    workspace: BaselineWorkspaceState
    workspace_fingerprints: ScreenWorkspaceFingerprints
    variant: FragmentVariantSpec
    selector_input_fields: list[str]
    configurations: list[FragmentRunArtifact] = Field(min_length=2, max_length=2)
    gates: list[FragmentGateResult] = Field(min_length=9, max_length=9)
    hybrid_context_token_ratio: float = Field(ge=0)
    valid: bool
    development_passed: bool
    confirmation_claim_allowed: Literal[False] = False


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> FragmentDevelopmentManifest:
    return FragmentDevelopmentManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def run_fragment_development(
    manifest: FragmentDevelopmentManifest,
    *,
    manifest_path: str | Path,
    repository_root: str | Path,
    workspace: LocalWorkspace,
    output_dir: str | Path,
    git_state: BaselineGitState,
    runtime_environment: BaselineRuntimeEnvironment,
) -> FragmentDevelopmentReport:
    root = Path(repository_root).resolve()
    source_manifest_path = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"E4b output directory already exists: {destination}")

    direction_manifest, validated = _validate_inputs(manifest, root, workspace)
    parent, parent_runs = _load_parent(manifest, root)
    fingerprints = _workspace_fingerprints(
        workspace,
        [file.path for file in direction_manifest.corpus.files],
    )
    if (
        fingerprints.chunk_content_sha256
        != manifest.workspace.expected_chunk_content_sha256
        or fingerprints.index_input_sha256
        != manifest.workspace.expected_index_input_sha256
    ):
        raise ValueError("E4b workspace fingerprint mismatch")

    chunk_records = workspace.read_chunks()
    chunk_by_id = _chunk_by_id(chunk_records)
    cases_by_id = {case.id: case for case in validated.cases}
    destination.mkdir(parents=True)
    atomic_write_text(
        destination / "manifest.json",
        manifest.model_dump_json(indent=2) + "\n",
    )

    configurations: list[FragmentRunArtifact] = []
    for mode in ("semantic", "hybrid"):
        run_spec, ranking_artifact = parent_runs[mode]
        artifact = build_fragment_run(
            manifest,
            mode=mode,
            ranking_artifact=ranking_artifact,
            parent_ranking_sha256=run_spec.sha256,
            cases_by_id=cases_by_id,
            chunk_by_id=chunk_by_id,
            git_commit=git_state.commit,
            workspace_snapshot_id=validated.workspace.snapshot_id,
        )
        atomic_write_text(
            destination / f"{mode}-k5.json",
            artifact.model_dump_json(indent=2) + "\n",
        )
        configurations.append(artifact)

    gates, hybrid_ratio = evaluate_fragment_gates(manifest, configurations)
    report = FragmentDevelopmentReport(
        benchmark=manifest.name,
        description=manifest.description,
        measured_at=datetime.now(UTC).isoformat(),
        manifest_path=_display_path(source_manifest_path, root),
        manifest_sha256=sha256_file(source_manifest_path, "text_lf"),
        git=git_state,
        runtime=runtime_environment,
        parent=parent,
        dataset=validated.dataset,
        corpus=validated.corpus,
        workspace=validated.workspace,
        workspace_fingerprints=fingerprints,
        variant=manifest.variant,
        selector_input_fields=FROZEN_SELECTOR_INPUT_FIELDS,
        configurations=configurations,
        gates=gates,
        hybrid_context_token_ratio=hybrid_ratio,
        valid=(
            validated.checks.passed
            and not git_state.dirty
            and all(
                item.selector_input_fields == FROZEN_SELECTOR_INPUT_FIELDS
                for item in configurations
            )
        ),
        development_passed=False,
    )
    report = report.model_copy(
        update={
            "development_passed": report.valid
            and all(gate.passed for gate in gates)
        }
    )
    atomic_write_text(
        destination / "summary.json",
        report.model_dump_json(indent=2) + "\n",
    )
    return report


def build_fragment_run(
    manifest: FragmentDevelopmentManifest,
    *,
    mode: FragmentMode,
    ranking_artifact: ScreenRunArtifact,
    parent_ranking_sha256: str,
    cases_by_id: Mapping[str, RetrievalEvalCase],
    chunk_by_id: Mapping[str, Mapping[str, object]],
    git_commit: str,
    workspace_snapshot_id: str,
    benchmark: str | None = None,
) -> FragmentRunArtifact:
    results = ranking_artifact.evaluation.results
    if not results or {result.id for result in results} != set(cases_by_id):
        raise ValueError("E4b parent ranking and dataset case sets differ")
    case_results = [
        _build_fragment_case(
            manifest,
            case=cases_by_id[result.id],
            parent_result=result,
            chunk_by_id=chunk_by_id,
        )
        for result in results
    ]
    metrics = _aggregate_metrics(case_results)
    return FragmentRunArtifact(
        benchmark=benchmark or manifest.name,
        variant_id=manifest.variant.id,
        git_commit=git_commit,
        workspace_snapshot_id=workspace_snapshot_id,
        retrieval_mode=mode,
        parent_ranking_sha256=parent_ranking_sha256,
        parent_result_fingerprint_sha256=result_fingerprint(results),
        selector_input_fields=FROZEN_SELECTOR_INPUT_FIELDS,
        oracle=FragmentOracleMetadata(),
        metrics=metrics,
        cases=case_results,
    )


def evaluate_fragment_gates(
    manifest: FragmentDevelopmentManifest,
    configurations: Sequence[FragmentRunArtifact],
) -> tuple[list[FragmentGateResult], float]:
    by_mode = {item.retrieval_mode: item for item in configurations}
    if set(by_mode) != {"semantic", "hybrid"}:
        raise ValueError("E4b requires Semantic and Hybrid configurations")
    semantic = by_mode["semantic"]
    hybrid = by_mode["hybrid"]
    promotion = manifest.promotion
    unscorable = sorted(
        f"{item.retrieval_mode}:{case_id}"
        for item in configurations
        for case_id in item.metrics.unscorable_parent_hit_case_ids
    )
    oracle_losses = sorted(
        f"{item.retrieval_mode}:{case_id}"
        for item in configurations
        for case_id in item.metrics.oracle_lost_hit_case_ids
    )
    candidate_violations = _mode_violations(
        configurations,
        "candidate_representation_violations",
    )
    traceability_violations = _mode_violations(
        configurations,
        "traceability_violations",
    )
    budget_violations = _mode_violations(configurations, "budget_violations")
    mapping_violations = _mode_violations(configurations, "mapping_violations")
    hybrid_ratio = round_metric(
        hybrid.metrics.average_selector_context_tokens
        / manifest.e0_hybrid_top5_average_context_tokens
    )
    isolation_valid = (
        manifest.variant.selector_input_fields == FROZEN_SELECTOR_INPUT_FIELDS
        and all(
            item.selector_input_fields == FROZEN_SELECTOR_INPUT_FIELDS
            for item in configurations
        )
    )
    return (
        [
            FragmentGateResult(
                name="oracle_evidence_reachable",
                passed=(
                    len(oracle_losses) <= promotion.max_oracle_hit_losses
                    and len(unscorable) <= promotion.max_unscorable_parent_hits
                ),
                observed=(
                    f"losses={len(oracle_losses)} unscorable={len(unscorable)}"
                ),
                requirement=(
                    f"losses<={promotion.max_oracle_hit_losses} "
                    f"unscorable<={promotion.max_unscorable_parent_hits}"
                ),
                case_ids=[*oracle_losses, *unscorable],
            ),
            FragmentGateResult(
                name="semantic_fragment_hits_retained",
                passed=(
                    len(semantic.metrics.selector_lost_hit_case_ids)
                    <= promotion.max_semantic_fragment_hit_losses
                ),
                observed=(
                    f"losses={len(semantic.metrics.selector_lost_hit_case_ids)}"
                ),
                requirement=(
                    f"losses<={promotion.max_semantic_fragment_hit_losses}"
                ),
                case_ids=semantic.metrics.selector_lost_hit_case_ids,
            ),
            FragmentGateResult(
                name="hybrid_fragment_hits_retained",
                passed=(
                    len(hybrid.metrics.selector_lost_hit_case_ids)
                    <= promotion.max_hybrid_fragment_hit_losses
                ),
                observed=f"losses={len(hybrid.metrics.selector_lost_hit_case_ids)}",
                requirement=f"losses<={promotion.max_hybrid_fragment_hit_losses}",
                case_ids=hybrid.metrics.selector_lost_hit_case_ids,
            ),
            FragmentGateResult(
                name="all_candidates_represented",
                passed=not candidate_violations,
                observed=f"violations={len(candidate_violations)}",
                requirement="violations=0",
                case_ids=candidate_violations,
            ),
            FragmentGateResult(
                name="all_fragments_traceable",
                passed=not traceability_violations,
                observed=f"violations={len(traceability_violations)}",
                requirement="violations=0",
                case_ids=traceability_violations,
            ),
            FragmentGateResult(
                name="all_contexts_within_budget",
                passed=not budget_violations,
                observed=f"violations={len(budget_violations)}",
                requirement="violations=0",
                case_ids=budget_violations,
            ),
            FragmentGateResult(
                name="complete_evidence_mapping",
                passed=not mapping_violations,
                observed=f"violations={len(mapping_violations)}",
                requirement="violations=0",
                case_ids=mapping_violations,
            ),
            FragmentGateResult(
                name="hybrid_context_tokens",
                passed=hybrid_ratio <= promotion.max_hybrid_context_token_ratio,
                observed=f"ratio={hybrid_ratio:.4f}",
                requirement=(
                    f"ratio<={promotion.max_hybrid_context_token_ratio:.4f}"
                ),
            ),
            FragmentGateResult(
                name="selector_gold_isolation",
                passed=isolation_valid,
                observed=f"allowlist_match={str(isolation_valid).lower()}",
                requirement="allowlist_match=true",
            ),
        ],
        hybrid_ratio,
    )


def _build_fragment_case(
    manifest: FragmentDevelopmentManifest,
    *,
    case: RetrievalEvalCase,
    parent_result: RetrievalEvalCaseResult,
    chunk_by_id: Mapping[str, Mapping[str, object]],
) -> FragmentCaseResult:
    candidates = _ranked_candidates(parent_result, chunk_by_id)
    gold_text = "\n".join(span.text for span in case.evidence_spans)
    oracle_fragments = select_ranked_fragments(
        candidates,
        max_context_chars=manifest.variant.max_context_chars,
        max_fragment_chars=manifest.variant.max_fragment_chars,
        stride_chars=manifest.variant.stride_chars,
        scorer=build_evidence_window_scorer(
            gold_text,
            ngram_size=manifest.variant.evidence_ngram_size,
        ),
    )
    selector_fragments = select_ranked_fragments(
        candidates,
        max_context_chars=manifest.variant.max_context_chars,
        max_fragment_chars=manifest.variant.max_fragment_chars,
        stride_chars=manifest.variant.stride_chars,
        scorer=build_query_window_scorer(case.query),
    )
    ranked_ids = [candidate.chunk_id for candidate in candidates]
    relevant_ids = set(parent_result.expected_chunk_ids)
    parent_hit = bool(relevant_ids & set(ranked_ids))
    evidence = _fragment_evidence(
        gold_text,
        candidates=candidates,
        oracle_fragments=oracle_fragments,
        selector_fragments=selector_fragments,
        relevant_chunk_ids=relevant_ids,
        parent_hit=parent_hit,
        ngram_size=manifest.variant.evidence_ngram_size,
    )
    oracle_rendered = render_fragments(oracle_fragments)
    selector_rendered = render_fragments(selector_fragments)
    text_by_id = {candidate.chunk_id: candidate.text for candidate in candidates}
    represented = (
        [fragment.chunk_id for fragment in oracle_fragments] == ranked_ids
        and [fragment.chunk_id for fragment in selector_fragments] == ranked_ids
    )
    traceable = fragments_are_traceable(oracle_fragments, text_by_id) and (
        fragments_are_traceable(selector_fragments, text_by_id)
    )
    max_chars = manifest.variant.max_context_chars
    budget_respected = (
        bool(oracle_rendered)
        and bool(selector_rendered)
        and len(oracle_rendered) <= max_chars
        and len(selector_rendered) <= max_chars
    )
    mapping_coverage = parent_result.mapping_coverage
    if mapping_coverage is None:
        raise ValueError(f"E4b parent mapping coverage missing: {case.id}")
    return FragmentCaseResult(
        case_id=case.id,
        query=case.query,
        parent_relevant_ranks=parent_result.relevant_result_ranks,
        parent_relevant_chunk_ids=parent_result.expected_chunk_ids,
        parent_mapping_coverage=mapping_coverage,
        ranked_chunk_ids=ranked_ids,
        oracle_fragments=[_fragment_record(item) for item in oracle_fragments],
        selector_fragments=[_fragment_record(item) for item in selector_fragments],
        oracle_rendered_context=oracle_rendered,
        selector_rendered_context=selector_rendered,
        oracle_estimated_context_tokens=math.ceil(
            len(oracle_rendered) / manifest.variant.characters_per_token
        ),
        selector_estimated_context_tokens=math.ceil(
            len(selector_rendered) / manifest.variant.characters_per_token
        ),
        candidates_represented=represented,
        fragments_traceable=traceable,
        budget_respected=budget_respected,
        evidence=evidence,
    )


def _fragment_evidence(
    gold_text: str,
    *,
    candidates: Sequence[RankedFragmentCandidate],
    oracle_fragments: Sequence[ContextFragment],
    selector_fragments: Sequence[ContextFragment],
    relevant_chunk_ids: set[str],
    parent_hit: bool,
    ngram_size: int,
) -> FragmentEvidenceMetrics:
    gold = normalized_token_ngrams(gold_text, ngram_size=ngram_size)
    reachable = gold & _ngram_union(
        candidate.text
        for candidate in candidates
        if candidate.chunk_id in relevant_chunk_ids
    )
    oracle = gold & _ngram_union(
        fragment.text
        for fragment in oracle_fragments
        if fragment.chunk_id in relevant_chunk_ids
    )
    selected = gold & _ngram_union(
        fragment.text
        for fragment in selector_fragments
        if fragment.chunk_id in relevant_chunk_ids
    )
    scorable = parent_hit and bool(reachable)
    coverage = len(selected) / len(reachable) if reachable else None
    efficiency = len(selected) / len(oracle) if oracle else None
    return FragmentEvidenceMetrics(
        parent_hit=parent_hit,
        scorable=scorable,
        reachable_evidence_ngrams=len(reachable),
        selected_evidence_ngrams=len(selected),
        oracle_evidence_ngrams=len(oracle),
        reachable_evidence_coverage=(
            None if coverage is None else round_metric(coverage)
        ),
        oracle_efficiency=(
            None if efficiency is None else round_metric(efficiency)
        ),
        oracle_retained=(not parent_hit or bool(oracle)),
        selector_retained=(not parent_hit or bool(selected)),
    )


def _aggregate_metrics(cases: Sequence[FragmentCaseResult]) -> FragmentRunMetrics:
    if not cases:
        raise ValueError("E4b fragment aggregation requires at least one case")
    parent_hits = [case for case in cases if case.evidence.parent_hit]
    scorable = [case for case in parent_hits if case.evidence.scorable]
    coverages = [
        case.evidence.reachable_evidence_coverage
        for case in scorable
        if case.evidence.reachable_evidence_coverage is not None
    ]
    efficiencies = [
        case.evidence.oracle_efficiency
        for case in scorable
        if case.evidence.oracle_efficiency is not None
    ]
    return FragmentRunMetrics(
        case_count=len(cases),
        parent_hits=len(parent_hits),
        scorable_parent_hits=len(scorable),
        unscorable_parent_hit_case_ids=[
            case.case_id for case in parent_hits if not case.evidence.scorable
        ],
        oracle_lost_hit_case_ids=[
            case.case_id for case in parent_hits if not case.evidence.oracle_retained
        ],
        selector_lost_hit_case_ids=[
            case.case_id for case in parent_hits if not case.evidence.selector_retained
        ],
        average_selector_context_tokens=round_metric(
            sum(case.selector_estimated_context_tokens for case in cases) / len(cases)
        ),
        maximum_selector_context_tokens=max(
            case.selector_estimated_context_tokens for case in cases
        ),
        average_reachable_evidence_coverage=_optional_average(coverages),
        minimum_reachable_evidence_coverage=(min(coverages) if coverages else None),
        average_oracle_efficiency=_optional_average(efficiencies),
        candidate_representation_violations=[
            case.case_id for case in cases if not case.candidates_represented
        ],
        traceability_violations=[
            case.case_id for case in cases if not case.fragments_traceable
        ],
        budget_violations=[
            case.case_id for case in cases if not case.budget_respected
        ],
        mapping_violations=[
            case.case_id for case in cases if case.parent_mapping_coverage != 1.0
        ],
    )


def _validate_inputs(
    manifest: FragmentDevelopmentManifest,
    repository_root: Path,
    workspace: LocalWorkspace,
) -> tuple[DirectionConfirmationManifest, ValidatedBaselineInputs]:
    direction_path = _resolve_file(
        repository_root,
        manifest.parent.direction_manifest.path,
    )
    if (
        sha256_file(direction_path, manifest.parent.direction_manifest.hash_mode)
        != manifest.parent.direction_manifest.sha256
    ):
        raise ValueError("E4b direction manifest hash mismatch")
    direction = DirectionConfirmationManifest.model_validate_json(
        direction_path.read_text(encoding="utf-8")
    )
    validation_manifest = RetrievalBaselineManifest(
        name=manifest.name,
        description=manifest.description,
        dataset=BaselineDatasetSpec.model_validate(direction.dataset),
        corpus=BaselineCorpusSpec.model_validate(direction.corpus),
        ingest=BaselineIngestSpec.model_validate(direction.ingest),
        embedding=BaselineEmbeddingSpec.model_validate(direction.embedding),
        workload=BaselineWorkloadSpec(
            retrieval_modes=["semantic", "hybrid"],
            limits=[5],
            repetitions=3,
            max_quality_metric_spread=0.05,
        ),
    )
    validated = validate_inputs(
        validation_manifest,
        repository_root,
        workspace,
        manifest.workspace.build_git_commit,
    )
    if len(validated.cases) != 50 or any(
        len(case.evidence_spans) != 1 for case in validated.cases
    ):
        raise ValueError("E4b requires 50 cases with one evidence span each")
    return direction, validated


def _load_parent(
    manifest: FragmentDevelopmentManifest,
    repository_root: Path,
) -> tuple[
    ResolvedFragmentParent,
    dict[FragmentMode, tuple[FragmentParentRunSpec, ScreenRunArtifact]],
]:
    parent_path = _resolve_file(repository_root, manifest.parent.path)
    parent_sha = sha256_file(parent_path, manifest.parent.hash_mode)
    if parent_sha != manifest.parent.sha256:
        raise ValueError("E4b parent summary hash mismatch")
    summary = DirectionConfirmationReport.model_validate_json(
        parent_path.read_text(encoding="utf-8")
    )
    if (
        not summary.valid
        or summary.git.commit != manifest.parent.evaluation_git_commit
        or summary.workspace.snapshot_id != manifest.parent.workspace_snapshot_id
    ):
        raise ValueError("E4b parent summary provenance mismatch")
    resolved_runs: list[ResolvedFragmentParentRun] = []
    artifacts: dict[
        FragmentMode,
        tuple[FragmentParentRunSpec, ScreenRunArtifact],
    ] = {}
    for run_spec in manifest.parent.runs:
        run_path = _resolve_file(repository_root, run_spec.path)
        run_sha = sha256_file(run_path, run_spec.hash_mode)
        artifact = ScreenRunArtifact.model_validate_json(
            run_path.read_text(encoding="utf-8")
        )
        configuration = next(
            item
            for item in summary.configurations
            if item.retrieval_mode == run_spec.retrieval_mode
        )
        if (
            run_sha != run_spec.sha256
            or artifact.git_commit != manifest.parent.evaluation_git_commit
            or artifact.workspace_snapshot_id != manifest.parent.workspace_snapshot_id
            or artifact.evaluation.retrieval_mode != run_spec.retrieval_mode
            or artifact.evaluation.limit != 5
            or len(artifact.evaluation.results) != 50
            or result_fingerprint(artifact.evaluation.results)
            != run_spec.result_fingerprint_sha256
            or configuration.result_fingerprint_sha256
            != run_spec.result_fingerprint_sha256
        ):
            raise ValueError(f"E4b parent run provenance mismatch: {run_spec.path}")
        resolved_runs.append(
            ResolvedFragmentParentRun(
                retrieval_mode=run_spec.retrieval_mode,
                file=BaselineResolvedFile(
                    path=_display_path(run_path, repository_root),
                    sha256=run_sha,
                    hash_mode=run_spec.hash_mode,
                ),
                result_fingerprint_sha256=run_spec.result_fingerprint_sha256,
            )
        )
        artifacts[run_spec.retrieval_mode] = (run_spec, artifact)
    direction_path = _resolve_file(
        repository_root,
        manifest.parent.direction_manifest.path,
    )
    return (
        ResolvedFragmentParent(
            summary=BaselineResolvedFile(
                path=_display_path(parent_path, repository_root),
                sha256=parent_sha,
                hash_mode=manifest.parent.hash_mode,
            ),
            direction_manifest=BaselineResolvedFile(
                path=_display_path(direction_path, repository_root),
                sha256=manifest.parent.direction_manifest.sha256,
                hash_mode=manifest.parent.direction_manifest.hash_mode,
            ),
            evaluation_git_commit=manifest.parent.evaluation_git_commit,
            workspace_snapshot_id=manifest.parent.workspace_snapshot_id,
            runs=resolved_runs,
        ),
        artifacts,
    )


def _ranked_candidates(
    result: RetrievalEvalCaseResult,
    chunk_by_id: Mapping[str, Mapping[str, object]],
) -> list[RankedFragmentCandidate]:
    candidates: list[RankedFragmentCandidate] = []
    for expected_rank, item in enumerate(result.top_results, start=1):
        rank = item.get("rank")
        chunk_id = item.get("chunk_id")
        if rank != expected_rank or not isinstance(chunk_id, str):
            raise ValueError(f"E4b parent ranking shape mismatch: {result.id}")
        record = chunk_by_id.get(chunk_id)
        if record is None:
            raise ValueError(f"E4b parent chunk missing from workspace: {chunk_id}")
        text = record.get("text")
        source_path = record.get("source_path")
        if not isinstance(text, str) or not text:
            raise ValueError(f"E4b chunk text is invalid: {chunk_id}")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(f"E4b chunk source path is invalid: {chunk_id}")
        candidates.append(
            RankedFragmentCandidate(
                rank=expected_rank,
                chunk_id=chunk_id,
                source_path=source_path,
                source_label=_source_label(source_path),
                page_label=_page_label(record.get("metadata")),
                text=text,
                signal_text=_signal_text(record.get("metadata")),
            )
        )
    if len(candidates) != 5:
        raise ValueError(f"E4b parent ranking is not Top-5: {result.id}")
    return candidates


def _fragment_record(fragment: ContextFragment) -> FragmentRecord:
    return FragmentRecord(
        rank=fragment.rank,
        chunk_id=fragment.chunk_id,
        source_path=fragment.source_path,
        source_label=fragment.source_label,
        page_label=fragment.page_label,
        start_char=fragment.start_char,
        end_char=fragment.end_char,
        text=fragment.text,
        truncated_left=fragment.truncated_left,
        truncated_right=fragment.truncated_right,
        rendered_header=fragment.header,
        score=_score_record(fragment.score),
    )


def _score_record(score: FragmentScoreComponents) -> FragmentScoreRecord:
    return FragmentScoreRecord(
        unique_query_tokens=score.unique_query_tokens,
        query_token_occurrences=score.query_token_occurrences,
        query_bigram_coverage=score.query_bigram_coverage,
        signal_token_coverage=score.signal_token_coverage,
    )


def _chunk_by_id(
    chunks: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    by_id: dict[str, Mapping[str, object]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in by_id:
            raise ValueError("E4b workspace contains an invalid or duplicate chunk ID")
        by_id[chunk_id] = chunk
    return by_id


def _workspace_fingerprints(
    workspace: LocalWorkspace,
    corpus_paths: Sequence[str],
) -> ScreenWorkspaceFingerprints:
    chunks = workspace.read_chunks()
    records = VectorIndexService(workspace).read_index()
    return ScreenWorkspaceFingerprints(
        chunk_content_sha256=chunk_content_fingerprint(chunks, corpus_paths),
        index_input_sha256=index_input_fingerprint(records, corpus_paths),
    )


def _ngram_union(texts: Iterable[str]) -> set[tuple[str, ...]]:
    ngrams: set[tuple[str, ...]] = set()
    for text in texts:
        ngrams.update(normalized_token_ngrams(text, ngram_size=3))
    return ngrams


def _optional_average(values: Sequence[float]) -> float | None:
    return round_metric(sum(values) / len(values)) if values else None


def _mode_violations(
    configurations: Sequence[FragmentRunArtifact],
    field: Literal[
        "candidate_representation_violations",
        "traceability_violations",
        "budget_violations",
        "mapping_violations",
    ],
) -> list[str]:
    return sorted(
        f"{item.retrieval_mode}:{case_id}"
        for item in configurations
        for case_id in getattr(item.metrics, field)
    )


def _signal_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    strings: list[str] = []
    for key in ("section_title", "heading_path", "possible_formula_lines"):
        item = value.get(key)
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, list):
            strings.extend(part for part in item if isinstance(part, str))
    return "\n".join(strings)


def _page_label(value: object) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    start = value.get("page_start")
    end = value.get("page_end")
    if not isinstance(start, int) or isinstance(start, bool):
        return "unknown"
    if isinstance(end, int) and not isinstance(end, bool) and end != start:
        return f"{start}-{end}"
    return str(start)


def _source_label(source_path: str) -> str:
    return source_path.replace("\\", "/").rsplit("/", 1)[-1]


def _resolve_file(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to(repository_root) or not path.is_file():
        raise FileNotFoundError(f"E4b input file is invalid: {value}")
    return path


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _print_summary(report: FragmentDevelopmentReport, output_dir: Path) -> None:
    print(f"E4b development: {report.benchmark}")
    for configuration in report.configurations:
        metrics = configuration.metrics
        print(
            f"{configuration.retrieval_mode}@5: parent={metrics.parent_hits} "
            f"scorable={metrics.scorable_parent_hits} "
            f"oracle-losses={len(metrics.oracle_lost_hit_case_ids)} "
            f"selector-losses={len(metrics.selector_lost_hit_case_ids)} "
            f"context-tokens={metrics.average_selector_context_tokens:.4f}"
        )
    print("Gates:")
    for gate in report.gates:
        print(
            f"  {'PASS' if gate.passed else 'FAIL'} {gate.name}: "
            f"{gate.observed} ({gate.requirement})"
        )
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Valid: {report.valid}")
    print(f"Development passed: {report.development_passed}")
    print("Confirmation claim allowed: False")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay frozen Top-5 rankings through E4b fragment packing."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = Path.cwd().resolve()
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(completed.stdout.strip()).resolve()
        git_state = collect_git_state(root)
        if git_state.dirty and not args.allow_dirty:
            raise ValueError("E4b development requires a clean Git tree")
        report = run_fragment_development(
            load_manifest(args.manifest),
            manifest_path=args.manifest,
            repository_root=root,
            workspace=LocalWorkspace(args.workspace),
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
        print(f"E4b development failed: {exc}", file=sys.stderr)
        return 1
    _print_summary(report, Path(args.output_dir).resolve())
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
