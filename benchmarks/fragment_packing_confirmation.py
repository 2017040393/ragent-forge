from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from benchmarks.finalize_e4b_heldout import FinalizedHeldoutDatasetManifest
from benchmarks.fragment_packing_development import (
    FROZEN_SELECTOR_INPUT_FIELDS,
    FragmentDevelopmentManifest,
    FragmentDevelopmentReport,
    FragmentMode,
    FragmentRunArtifact,
    FragmentVariantSpec,
    _chunk_by_id,
    _workspace_fingerprints,
    build_fragment_run,
)
from benchmarks.generate_e4b_heldout import HeldoutGenerationReport
from benchmarks.retrieval_baseline import (
    ValidatedBaselineInputs,
    collect_git_state,
    collect_runtime_environment,
    sha256_file,
    validate_inputs,
)
from benchmarks.retrieval_screen import (
    CachedQueryEmbeddingService,
    QueryEmbeddingCacheFile,
)
from ragent_forge.app.ports import EmbeddingServicePort
from ragent_forge.app.services.evaluation.baseline import (
    BaselineCacheState,
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
from ragent_forge.app.services.evaluation.contracts import RetrievalEvalCase
from ragent_forge.app.services.evaluation.metrics import round_metric
from ragent_forge.app.services.evaluation.runner import RetrievalEvalService
from ragent_forge.app.services.evaluation.screening import (
    ScreenRunArtifact,
    ScreenWorkspaceFingerprints,
)
from ragent_forge.app.services.prepared_retrieval import PreparedStateCache
from ragent_forge.app.services.search_service import tokenize
from ragent_forge.composition import (
    build_embedding_service,
    build_retrieval_runtime,
)
from ragent_forge.core.retrieval.representations import (
    EmbeddingRepresentation,
    QueryEmbeddingRepresentation,
    build_query_embedding_text,
)
from ragent_forge.infrastructure.local_workspace import LocalWorkspace
from ragent_forge.infrastructure.storage import atomic_write_text

DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "fragment_packing_confirmation_manifest_e4b.json"
)

HeldoutGateName = Literal[
    "dataset_is_held_out",
    "minimum_parent_hits",
    "oracle_evidence_reachable",
    "fragment_hits_retained",
    "average_evidence_coverage",
    "minimum_evidence_coverage",
    "oracle_efficiency",
    "all_candidates_represented",
    "all_fragments_traceable",
    "all_contexts_within_budget",
    "complete_evidence_mapping",
    "hybrid_context_tokens",
    "selector_gold_isolation",
]

STRUCTURAL_GATE_NAMES: set[HeldoutGateName] = {
    "dataset_is_held_out",
    "all_candidates_represented",
    "all_fragments_traceable",
    "all_contexts_within_budget",
    "complete_evidence_mapping",
    "selector_gold_isolation",
}


class FrozenDevelopmentSpec(BaseModel):
    manifest: BaselineFileSpec
    summary: BaselineFileSpec
    implementation_git_commit: GitCommit
    workspace_snapshot_id: str = Field(min_length=1)


class HeldoutRankingSpec(BaseModel):
    document_embedding_representation: EmbeddingRepresentation
    query_embedding_representation: QueryEmbeddingRepresentation
    workspace_build_git_commit: GitCommit
    expected_chunk_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_index_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_limit: Literal[5] = 5

    @model_validator(mode="after")
    def _fixed_ranking_contract(self) -> HeldoutRankingSpec:
        if self.document_embedding_representation != (
            "cleaned_pdf_formula_text_v1"
        ):
            raise ValueError("E4b confirmation requires E3b document input")
        if self.query_embedding_representation != "instructed_query_v1":
            raise ValueError("E4b confirmation requires instructed queries")
        return self


class HeldoutConfirmationSpec(BaseModel):
    minimum_parent_hits_per_mode: Literal[8] = 8
    maximum_oracle_hit_losses: Literal[0] = 0
    maximum_fragment_hit_losses: Literal[0] = 0
    minimum_average_evidence_coverage: float = Field(ge=0, le=1)
    minimum_case_evidence_coverage: float = Field(ge=0, le=1)
    minimum_average_oracle_efficiency: float = Field(ge=0, le=1)
    required_candidates_per_case: Literal[5] = 5
    maximum_hybrid_context_token_ratio: float = Field(gt=0)

    @model_validator(mode="after")
    def _frozen_quality_thresholds(self) -> HeldoutConfirmationSpec:
        observed = (
            self.minimum_average_evidence_coverage,
            self.minimum_case_evidence_coverage,
            self.minimum_average_oracle_efficiency,
            self.maximum_hybrid_context_token_ratio,
        )
        if observed != (0.6, 0.25, 0.8, 1.1):
            raise ValueError("E4b held-out confirmation thresholds drifted")
        return self


class FragmentPackingConfirmationManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    development: FrozenDevelopmentSpec
    dataset: BaselineDatasetSpec
    canonical_dataset: BaselineFileSpec
    corpus: BaselineCorpusSpec
    ingest: BaselineIngestSpec
    embedding: BaselineEmbeddingSpec
    ranking: HeldoutRankingSpec
    variant: FragmentVariantSpec
    e0_hybrid_top5_average_context_tokens: float = Field(gt=0)
    confirmation: HeldoutConfirmationSpec

    @model_validator(mode="after")
    def _fixed_heldout_contract(self) -> FragmentPackingConfirmationManifest:
        if self.dataset.case_count != 20:
            raise ValueError("E4b held-out confirmation requires 20 cases")
        if self.e0_hybrid_top5_average_context_tokens != 724.5:
            raise ValueError("E4b E0 context-token reference drifted")
        if self.ranking.candidate_limit != self.variant.candidate_limit:
            raise ValueError("E4b ranking and fragment candidate limits differ")
        return self


class ResolvedFrozenDevelopment(BaseModel):
    manifest: BaselineResolvedFile
    summary: BaselineResolvedFile
    implementation_git_commit: GitCommit
    workspace_snapshot_id: str
    development_passed: bool


class HeldoutDatasetValidation(BaseModel):
    finalized_manifest: BaselineResolvedFile
    generation_summary: BaselineResolvedFile
    canonical_dataset: BaselineResolvedFile
    case_count: Literal[20] = 20
    unique_span_count: Literal[10] = 10
    canonical_query_duplicates: list[str]
    canonical_span_overlaps: list[str]
    provenance_violations: list[str]
    passed: bool


class HeldoutRankingCheckpoint(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    variant_id: str
    git_commit: GitCommit
    workspace_snapshot_id: str
    retrieval_mode: FragmentMode
    ranking_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HeldoutRankingReference(BaseModel):
    retrieval_mode: FragmentMode
    artifact: BaselineResolvedFile
    checkpoint: BaselineResolvedFile
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_hits: int = Field(ge=0, le=20)
    query_cache_hits: int = Field(ge=0)
    query_cache_misses: int = Field(ge=0)
    prepared_cache_reuse_valid: bool


class HeldoutQueryCacheReport(BaseModel):
    artifact: BaselineResolvedFile
    provider: str
    model: str
    query_representation: QueryEmbeddingRepresentation
    embedding_dim: int = Field(gt=0)
    entry_count: int = Field(ge=0)
    source_path: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    semantic_hits: int = Field(ge=0)
    semantic_misses: int = Field(ge=0)
    hybrid_hits: int = Field(ge=0)
    hybrid_misses: int = Field(ge=0)
    independent: bool
    hybrid_reuse_complete: bool
    valid: bool


class HeldoutObservationalSlices(BaseModel):
    question_type_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    source_counts: dict[str, int]
    evidence_page_counts: dict[str, int]
    relevant_rank_counts: dict[FragmentMode, dict[str, int]]


class HeldoutGateResult(BaseModel):
    name: HeldoutGateName
    passed: bool
    observed: str
    requirement: str
    case_ids: list[str] = Field(default_factory=list)


class FragmentPackingConfirmationReport(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    description: str
    measured_at: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git: BaselineGitState
    runtime: BaselineRuntimeEnvironment
    development: ResolvedFrozenDevelopment
    dataset: BaselineResolvedDataset
    dataset_holdout: HeldoutDatasetValidation
    corpus: list[BaselineResolvedFile]
    workspace: BaselineWorkspaceState
    workspace_fingerprints: ScreenWorkspaceFingerprints
    ranking: HeldoutRankingSpec
    variant: FragmentVariantSpec
    query_cache: HeldoutQueryCacheReport
    ranking_runs: list[HeldoutRankingReference] = Field(
        min_length=2, max_length=2
    )
    configurations: list[FragmentRunArtifact] = Field(
        min_length=2, max_length=2
    )
    observational_slices: HeldoutObservationalSlices
    gates: list[HeldoutGateResult] = Field(min_length=13, max_length=13)
    hybrid_context_token_ratio: float = Field(ge=0)
    valid: bool
    confirmed: bool


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> FragmentPackingConfirmationManifest:
    return FragmentPackingConfirmationManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def run_fragment_confirmation(
    manifest: FragmentPackingConfirmationManifest,
    *,
    manifest_path: str | Path,
    repository_root: str | Path,
    workspace: LocalWorkspace,
    output_dir: str | Path,
    git_state: BaselineGitState,
    runtime_environment: BaselineRuntimeEnvironment,
    embedding_service: EmbeddingServicePort | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> FragmentPackingConfirmationReport:
    root = Path(repository_root).resolve()
    source_manifest_path = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and not resume:
        raise FileExistsError(
            f"E4b held-out output directory already exists: {destination}"
        )
    if resume and not destination.is_dir():
        raise FileNotFoundError(
            f"E4b held-out resume directory is missing: {destination}"
        )

    development_manifest, development = _validate_development(manifest, root)
    validated = _validate_inputs(manifest, root, workspace)
    dataset_holdout = _validate_heldout_dataset(
        manifest,
        repository_root=root,
        cases=validated.cases,
    )
    fingerprints = _workspace_fingerprints(
        workspace,
        [file.path for file in manifest.corpus.files],
    )
    if (
        fingerprints.chunk_content_sha256
        != manifest.ranking.expected_chunk_content_sha256
        or fingerprints.index_input_sha256
        != manifest.ranking.expected_index_input_sha256
    ):
        raise ValueError("E4b held-out workspace fingerprint mismatch")

    if resume:
        _validate_resume_manifest(destination, manifest)
    else:
        destination.mkdir(parents=True)
        (destination / "ranking-runs").mkdir()
        (destination / "fragment-runs").mkdir()
        atomic_write_text(
            destination / "manifest.json",
            manifest.model_dump_json(indent=2) + "\n",
        )

    query_cache_path = destination / "query_embeddings.json"
    query_cache = CachedQueryEmbeddingService(
        embedding_service or build_embedding_service(validated.config),
        cache_path=query_cache_path,
        provider=manifest.embedding.provider,
        model=manifest.embedding.model,
        query_representation=manifest.ranking.query_embedding_representation,
        embedding_dim=manifest.embedding.dimensions,
        source_path=None,
        resume=resume,
    )
    prepared_cache = PreparedStateCache(tokenize)
    eval_service = RetrievalEvalService()
    cases_by_id = {case.id: case for case in validated.cases}
    chunk_by_id = _chunk_by_id(workspace.read_chunks())
    ranking_runs: list[HeldoutRankingReference] = []
    ranking_artifacts: dict[FragmentMode, ScreenRunArtifact] = {}
    configurations: list[FragmentRunArtifact] = []
    modes: tuple[FragmentMode, FragmentMode] = ("semantic", "hybrid")

    for mode in modes:
        ranking_relative = Path("ranking-runs") / f"{mode}-k5.json"
        ranking_path = destination / ranking_relative
        checkpoint_relative = (
            Path("ranking-runs") / f"{mode}-k5.checkpoint.json"
        )
        checkpoint_path = destination / checkpoint_relative
        if resume and ranking_path.is_file() and checkpoint_path.is_file():
            ranking_artifact = ScreenRunArtifact.model_validate_json(
                ranking_path.read_text(encoding="utf-8")
            )
            checkpoint = HeldoutRankingCheckpoint.model_validate_json(
                checkpoint_path.read_text(encoding="utf-8")
            )
            _validate_resume_ranking(
                ranking_artifact,
                checkpoint=checkpoint,
                manifest=manifest,
                mode=mode,
                git_commit=git_state.commit,
                workspace_snapshot_id=validated.workspace.snapshot_id,
                ranking_path=ranking_path,
                query_cache_path=query_cache_path,
                case_ids=set(cases_by_id),
            )
            if progress is not None:
                progress(f"Reused {mode}@5 ranking")
        else:
            if progress is not None:
                progress(f"Starting {mode}@5 ranking over 20 held-out cases")
            before = query_cache.stats()
            runtime = build_retrieval_runtime(
                workspace,
                mode,
                limit=manifest.ranking.candidate_limit,
                config=validated.config,
                prepared_state_cache=prepared_cache,
                embedding_service=query_cache,
            )
            evaluation = eval_service.evaluate(
                cases=validated.cases,
                search_service=runtime.retrieval_engine,
                limit=manifest.ranking.candidate_limit,
                retrieval_mode=mode,
                retrieval_method=runtime.retrieval_method,
                cases_path=manifest.dataset.path,
                workspace_path=workspace.root_path,
                embedding_provider=runtime.embedding_provider,
                embedding_model=runtime.embedding_model,
                index_path=runtime.index_path,
                fusion_method=runtime.fusion_method,
                rrf_k=runtime.rrf_k,
                sparse_method=runtime.sparse_method,
                dense_method=runtime.dense_method,
                sparse_weight=runtime.sparse_weight,
                dense_weight=runtime.dense_weight,
                lexical_weight=runtime.lexical_weight,
                semantic_weight=runtime.semantic_weight,
                workspace=workspace,
            )
            after = query_cache.stats()
            cache_state = _cache_state(prepared_cache)
            ranking_artifact = ScreenRunArtifact(
                benchmark=manifest.name,
                variant_id=manifest.variant.id,
                git_commit=git_state.commit,
                workspace_snapshot_id=validated.workspace.snapshot_id,
                cache=cache_state,
                cache_reuse_valid=_cache_valid(cache_state),
                query_cache_hits=after.hits - before.hits,
                query_cache_misses=after.misses - before.misses,
                evaluation=evaluation,
            )
            atomic_write_text(
                ranking_path,
                ranking_artifact.model_dump_json(indent=2) + "\n",
            )
            checkpoint = _build_ranking_checkpoint(
                ranking_artifact,
                manifest=manifest,
                mode=mode,
                ranking_path=ranking_path,
                query_cache_path=query_cache_path,
            )
            atomic_write_text(
                checkpoint_path,
                checkpoint.model_dump_json(indent=2) + "\n",
            )
            if progress is not None:
                hits = sum(
                    bool(result.relevant_result_ranks)
                    for result in evaluation.results
                )
                progress(f"Completed {mode}@5 ranking: parent-hits={hits}/20")

        if not _ranking_artifact_valid(
            ranking_artifact,
            manifest=manifest,
            mode=mode,
            case_ids=set(cases_by_id),
        ):
            raise ValueError(f"E4b held-out ranking artifact is invalid: {mode}")
        ranking_artifacts[mode] = ranking_artifact
        ranking_runs.append(
            _ranking_reference(
                ranking_artifact,
                mode=mode,
                ranking_path=ranking_path,
                checkpoint_path=checkpoint_path,
                root=destination,
            )
        )

        fragment_artifact = build_fragment_run(
            development_manifest,
            mode=mode,
            ranking_artifact=ranking_artifact,
            parent_ranking_sha256=sha256_file(ranking_path, "text_lf"),
            cases_by_id=cases_by_id,
            chunk_by_id=chunk_by_id,
            git_commit=git_state.commit,
            workspace_snapshot_id=validated.workspace.snapshot_id,
            benchmark=manifest.name,
        )
        atomic_write_text(
            destination / "fragment-runs" / f"{mode}-k5.json",
            fragment_artifact.model_dump_json(indent=2) + "\n",
        )
        configurations.append(fragment_artifact)

    query_cache_report = _query_cache_report(
        manifest,
        cache_path=query_cache_path,
        cache=QueryEmbeddingCacheFile.model_validate_json(
            query_cache_path.read_text(encoding="utf-8")
        ),
        cases=validated.cases,
        ranking_artifacts=ranking_artifacts,
    )
    gates, hybrid_ratio = evaluate_confirmation_gates(
        manifest,
        configurations,
        dataset_holdout=dataset_holdout,
    )
    structural_valid = all(
        gate.passed for gate in gates if gate.name in STRUCTURAL_GATE_NAMES
    )
    report = FragmentPackingConfirmationReport(
        benchmark=manifest.name,
        description=manifest.description,
        measured_at=datetime.now(UTC).isoformat(),
        manifest_path=_display_path(source_manifest_path, root),
        manifest_sha256=sha256_file(source_manifest_path, "text_lf"),
        git=git_state,
        runtime=runtime_environment,
        development=development,
        dataset=validated.dataset,
        dataset_holdout=dataset_holdout,
        corpus=validated.corpus,
        workspace=validated.workspace,
        workspace_fingerprints=fingerprints,
        ranking=manifest.ranking,
        variant=manifest.variant,
        query_cache=query_cache_report,
        ranking_runs=ranking_runs,
        configurations=configurations,
        observational_slices=_observational_slices(
            validated.cases, configurations
        ),
        gates=gates,
        hybrid_context_token_ratio=hybrid_ratio,
        valid=(
            validated.checks.passed
            and not git_state.dirty
            and development.development_passed
            and dataset_holdout.passed
            and query_cache_report.valid
            and len(ranking_artifacts) == 2
            and structural_valid
        ),
        confirmed=False,
    )
    report = report.model_copy(
        update={
            "confirmed": report.valid and all(gate.passed for gate in gates)
        }
    )
    atomic_write_text(
        destination / "summary.json",
        report.model_dump_json(indent=2) + "\n",
    )
    return report


def evaluate_confirmation_gates(
    manifest: FragmentPackingConfirmationManifest,
    configurations: Sequence[FragmentRunArtifact],
    *,
    dataset_holdout: HeldoutDatasetValidation,
) -> tuple[list[HeldoutGateResult], float]:
    by_mode = {item.retrieval_mode: item for item in configurations}
    if set(by_mode) != {"semantic", "hybrid"}:
        raise ValueError("E4b held-out requires Semantic and Hybrid runs")
    hybrid = by_mode["hybrid"]
    spec = manifest.confirmation

    parent_hit_failures = sorted(
        f"{mode}:{case.case_id}"
        for mode, run in by_mode.items()
        if run.metrics.parent_hits < spec.minimum_parent_hits_per_mode
        for case in run.cases
        if not case.evidence.parent_hit
    )
    unscorable = _mode_case_ids(
        configurations, "unscorable_parent_hit_case_ids"
    )
    oracle_losses = _mode_case_ids(
        configurations, "oracle_lost_hit_case_ids"
    )
    fragment_losses = _mode_case_ids(
        configurations, "selector_lost_hit_case_ids"
    )
    average_coverage_failures = [
        mode
        for mode, run in by_mode.items()
        if run.metrics.average_reachable_evidence_coverage is None
        or run.metrics.average_reachable_evidence_coverage
        < spec.minimum_average_evidence_coverage
    ]
    minimum_coverage_failures = sorted(
        f"{mode}:{case.case_id}"
        for mode, run in by_mode.items()
        for case in run.cases
        if case.evidence.scorable
        and (
            case.evidence.reachable_evidence_coverage is None
            or case.evidence.reachable_evidence_coverage
            < spec.minimum_case_evidence_coverage
        )
    )
    efficiency_failures = [
        mode
        for mode, run in by_mode.items()
        if run.metrics.average_oracle_efficiency is None
        or run.metrics.average_oracle_efficiency
        < spec.minimum_average_oracle_efficiency
    ]
    candidate_violations = _mode_case_ids(
        configurations, "candidate_representation_violations"
    )
    traceability_violations = _mode_case_ids(
        configurations, "traceability_violations"
    )
    budget_violations = _mode_case_ids(configurations, "budget_violations")
    mapping_violations = _mode_case_ids(configurations, "mapping_violations")
    hybrid_ratio = round_metric(
        hybrid.metrics.average_selector_context_tokens
        / manifest.e0_hybrid_top5_average_context_tokens
    )
    isolation_valid = (
        manifest.variant.selector_input_fields == FROZEN_SELECTOR_INPUT_FIELDS
        and all(
            run.selector_input_fields == FROZEN_SELECTOR_INPUT_FIELDS
            for run in configurations
        )
    )
    parent_observed = " ".join(
        f"{mode}={run.metrics.parent_hits}"
        for mode, run in by_mode.items()
    )
    average_coverage_observed = " ".join(
        f"{mode}={_format_optional(run.metrics.average_reachable_evidence_coverage)}"
        for mode, run in by_mode.items()
    )
    minimum_coverage_observed = " ".join(
        f"{mode}={_format_optional(run.metrics.minimum_reachable_evidence_coverage)}"
        for mode, run in by_mode.items()
    )
    efficiency_observed = " ".join(
        f"{mode}={_format_optional(run.metrics.average_oracle_efficiency)}"
        for mode, run in by_mode.items()
    )
    dataset_violations = [
        *dataset_holdout.canonical_query_duplicates,
        *dataset_holdout.canonical_span_overlaps,
        *dataset_holdout.provenance_violations,
    ]
    return (
        [
            HeldoutGateResult(
                name="dataset_is_held_out",
                passed=dataset_holdout.passed,
                observed=f"violations={len(dataset_violations)}",
                requirement="violations=0",
                case_ids=dataset_violations,
            ),
            HeldoutGateResult(
                name="minimum_parent_hits",
                passed=all(
                    run.metrics.parent_hits
                    >= spec.minimum_parent_hits_per_mode
                    for run in configurations
                ),
                observed=parent_observed,
                requirement=(
                    f"each>={spec.minimum_parent_hits_per_mode}"
                ),
                case_ids=parent_hit_failures,
            ),
            HeldoutGateResult(
                name="oracle_evidence_reachable",
                passed=(
                    len(unscorable) == 0
                    and len(oracle_losses)
                    <= spec.maximum_oracle_hit_losses
                ),
                observed=(
                    f"losses={len(oracle_losses)} "
                    f"unscorable={len(unscorable)}"
                ),
                requirement=(
                    f"losses<={spec.maximum_oracle_hit_losses} unscorable=0"
                ),
                case_ids=[*oracle_losses, *unscorable],
            ),
            HeldoutGateResult(
                name="fragment_hits_retained",
                passed=(
                    len(fragment_losses)
                    <= spec.maximum_fragment_hit_losses
                ),
                observed=f"losses={len(fragment_losses)}",
                requirement=(
                    f"losses<={spec.maximum_fragment_hit_losses}"
                ),
                case_ids=fragment_losses,
            ),
            HeldoutGateResult(
                name="average_evidence_coverage",
                passed=not average_coverage_failures,
                observed=average_coverage_observed,
                requirement=(
                    f"each>={spec.minimum_average_evidence_coverage:.4f}"
                ),
                case_ids=average_coverage_failures,
            ),
            HeldoutGateResult(
                name="minimum_evidence_coverage",
                passed=not minimum_coverage_failures,
                observed=minimum_coverage_observed,
                requirement=(
                    f"every_parent_hit>={spec.minimum_case_evidence_coverage:.4f}"
                ),
                case_ids=minimum_coverage_failures,
            ),
            HeldoutGateResult(
                name="oracle_efficiency",
                passed=not efficiency_failures,
                observed=efficiency_observed,
                requirement=(
                    f"each_average>={spec.minimum_average_oracle_efficiency:.4f}"
                ),
                case_ids=efficiency_failures,
            ),
            HeldoutGateResult(
                name="all_candidates_represented",
                passed=not candidate_violations,
                observed=f"violations={len(candidate_violations)}",
                requirement="violations=0",
                case_ids=candidate_violations,
            ),
            HeldoutGateResult(
                name="all_fragments_traceable",
                passed=not traceability_violations,
                observed=f"violations={len(traceability_violations)}",
                requirement="violations=0",
                case_ids=traceability_violations,
            ),
            HeldoutGateResult(
                name="all_contexts_within_budget",
                passed=not budget_violations,
                observed=f"violations={len(budget_violations)}",
                requirement="violations=0",
                case_ids=budget_violations,
            ),
            HeldoutGateResult(
                name="complete_evidence_mapping",
                passed=not mapping_violations,
                observed=f"violations={len(mapping_violations)}",
                requirement="violations=0",
                case_ids=mapping_violations,
            ),
            HeldoutGateResult(
                name="hybrid_context_tokens",
                passed=(
                    hybrid_ratio
                    <= spec.maximum_hybrid_context_token_ratio
                ),
                observed=f"ratio={hybrid_ratio:.4f}",
                requirement=(
                    "ratio<="
                    f"{spec.maximum_hybrid_context_token_ratio:.4f}"
                ),
            ),
            HeldoutGateResult(
                name="selector_gold_isolation",
                passed=isolation_valid,
                observed=f"allowlist_match={str(isolation_valid).lower()}",
                requirement="allowlist_match=true",
            ),
        ],
        hybrid_ratio,
    )


def _validate_development(
    manifest: FragmentPackingConfirmationManifest,
    root: Path,
) -> tuple[FragmentDevelopmentManifest, ResolvedFrozenDevelopment]:
    manifest_path = _validate_file_spec(manifest.development.manifest, root)
    summary_path = _validate_file_spec(manifest.development.summary, root)
    development_manifest = FragmentDevelopmentManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    summary = FragmentDevelopmentReport.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    if (
        summary.manifest_sha256 != manifest.development.manifest.sha256
        or summary.git.commit != manifest.development.implementation_git_commit
        or summary.git.dirty
        or not summary.valid
        or not summary.development_passed
        or summary.confirmation_claim_allowed
        or summary.workspace.snapshot_id
        != manifest.development.workspace_snapshot_id
        or development_manifest.variant != manifest.variant
        or summary.variant != manifest.variant
        or summary.workspace_fingerprints.chunk_content_sha256
        != manifest.ranking.expected_chunk_content_sha256
        or summary.workspace_fingerprints.index_input_sha256
        != manifest.ranking.expected_index_input_sha256
    ):
        raise ValueError("E4b frozen development provenance mismatch")
    return (
        development_manifest,
        ResolvedFrozenDevelopment(
            manifest=BaselineResolvedFile(
                path=_display_path(manifest_path, root),
                sha256=manifest.development.manifest.sha256,
                hash_mode=manifest.development.manifest.hash_mode,
            ),
            summary=BaselineResolvedFile(
                path=_display_path(summary_path, root),
                sha256=manifest.development.summary.sha256,
                hash_mode=manifest.development.summary.hash_mode,
            ),
            implementation_git_commit=(
                manifest.development.implementation_git_commit
            ),
            workspace_snapshot_id=manifest.development.workspace_snapshot_id,
            development_passed=True,
        ),
    )


def _validate_inputs(
    manifest: FragmentPackingConfirmationManifest,
    root: Path,
    workspace: LocalWorkspace,
) -> ValidatedBaselineInputs:
    validation_manifest = RetrievalBaselineManifest(
        name=manifest.name,
        description=manifest.description,
        dataset=manifest.dataset,
        corpus=manifest.corpus,
        ingest=manifest.ingest,
        embedding=manifest.embedding,
        workload=BaselineWorkloadSpec(
            retrieval_modes=["semantic", "hybrid"],
            limits=[5],
            repetitions=3,
            max_quality_metric_spread=0.05,
        ),
    )
    validated = validate_inputs(
        validation_manifest,
        root,
        workspace,
        manifest.ranking.workspace_build_git_commit,
    )
    if (
        len(validated.cases) != 20
        or any(len(case.evidence_spans) != 1 for case in validated.cases)
        or validated.workspace.snapshot_id
        != manifest.development.workspace_snapshot_id
    ):
        raise ValueError("E4b held-out dataset or workspace shape mismatch")
    return validated


def _validate_heldout_dataset(
    manifest: FragmentPackingConfirmationManifest,
    *,
    repository_root: Path,
    cases: Sequence[RetrievalEvalCase],
) -> HeldoutDatasetValidation:
    finalized_path = _validate_file_spec(manifest.dataset.manifest, repository_root)
    finalized = FinalizedHeldoutDatasetManifest.model_validate_json(
        finalized_path.read_text(encoding="utf-8")
    )
    generation_path = _validate_file_spec(
        finalized.generation_summary, repository_root
    )
    _validate_file_spec(finalized.raw_dataset, repository_root)
    _validate_file_spec(finalized.review_manifest, repository_root)
    canonical_path = _validate_file_spec(
        manifest.canonical_dataset, repository_root
    )
    if finalized.canonical_dataset != manifest.canonical_dataset:
        raise ValueError("E4b finalized canonical dataset provenance mismatch")
    generation = HeldoutGenerationReport.model_validate_json(
        generation_path.read_text(encoding="utf-8")
    )
    for resolved in generation.resolved_spans:
        artifact_path = (generation_path.parent / resolved.artifact_path).resolve()
        if not artifact_path.is_relative_to(generation_path.parent.resolve()):
            raise ValueError("E4b held-out span artifact escapes archive")
        if sha256_file(artifact_path, "text_lf") != resolved.artifact_sha256:
            raise ValueError(f"E4b held-out span artifact drifted: {resolved.span_id}")

    canonical_cases = _load_cases(canonical_path)
    canonical_queries = {
        case.query.strip().casefold() for case in canonical_cases
    }
    canonical_span_ids = {
        span.id for case in canonical_cases for span in case.evidence_spans
    }
    queries = [case.query.strip().casefold() for case in cases]
    canonical_query_duplicates = sorted(set(queries) & canonical_queries)
    actual_spans = [case.evidence_spans[0] for case in cases]
    span_counts = Counter(span.id for span in actual_spans)
    canonical_span_overlaps = sorted(set(span_counts) & canonical_span_ids)
    expected_spans = {span.span_id: span for span in generation.resolved_spans}
    provenance_violations: list[str] = []
    if not finalized.valid:
        provenance_violations.append("finalized_manifest_invalid")
    if not generation.valid:
        provenance_violations.append("generation_report_invalid")
    if finalized.dataset_path != manifest.dataset.path:
        provenance_violations.append("finalized_dataset_path")
    if finalized.dataset_sha256 != manifest.dataset.sha256:
        provenance_violations.append("finalized_dataset_hash")
    if set(span_counts) != set(expected_spans):
        provenance_violations.append("heldout_span_set")
    if any(count != 2 for count in span_counts.values()):
        provenance_violations.append("heldout_span_case_count")
    if len(set(queries)) != 20:
        provenance_violations.append("heldout_query_uniqueness")
    for span in actual_spans:
        expected = expected_spans.get(span.id)
        digest = hashlib.sha256(span.text.encode("utf-8")).hexdigest()
        if expected is None:
            continue
        metadata_digest = span.metadata.get("text_sha256")
        if (
            span.source_path != expected.source_path
            or span.page_start != expected.page
            or len(span.text) != expected.text_chars
            or digest != expected.text_sha256
            or metadata_digest != expected.text_sha256
        ):
            provenance_violations.append(span.id)
    provenance_violations = sorted(set(provenance_violations))
    passed = (
        not canonical_query_duplicates
        and not canonical_span_overlaps
        and not provenance_violations
        and len(span_counts) == 10
    )
    return HeldoutDatasetValidation(
        finalized_manifest=BaselineResolvedFile(
            path=_display_path(finalized_path, repository_root),
            sha256=manifest.dataset.manifest.sha256,
            hash_mode=manifest.dataset.manifest.hash_mode,
        ),
        generation_summary=BaselineResolvedFile(
            path=_display_path(generation_path, repository_root),
            sha256=finalized.generation_summary.sha256,
            hash_mode=finalized.generation_summary.hash_mode,
        ),
        canonical_dataset=BaselineResolvedFile(
            path=_display_path(canonical_path, repository_root),
            sha256=manifest.canonical_dataset.sha256,
            hash_mode=manifest.canonical_dataset.hash_mode,
        ),
        canonical_query_duplicates=canonical_query_duplicates,
        canonical_span_overlaps=canonical_span_overlaps,
        provenance_violations=provenance_violations,
        passed=passed,
    )


def _query_cache_report(
    manifest: FragmentPackingConfirmationManifest,
    *,
    cache_path: Path,
    cache: QueryEmbeddingCacheFile,
    cases: Sequence[RetrievalEvalCase],
    ranking_artifacts: Mapping[FragmentMode, ScreenRunArtifact],
) -> HeldoutQueryCacheReport:
    required_keys = {
        _query_key(
            build_query_embedding_text(
                case.query,
                manifest.ranking.query_embedding_representation,
            )
        )
        for case in cases
    }
    semantic = ranking_artifacts["semantic"]
    hybrid = ranking_artifacts["hybrid"]
    independent = cache.source_path is None and cache.source_sha256 is None
    hybrid_reuse_complete = (
        hybrid.query_cache_hits == 20 and hybrid.query_cache_misses == 0
    )
    valid = (
        independent
        and set(cache.entries) == required_keys
        and len(cache.entries) == 20
        and cache.provider == manifest.embedding.provider
        and cache.model == manifest.embedding.model
        and cache.query_representation
        == manifest.ranking.query_embedding_representation
        and cache.embedding_dim == manifest.embedding.dimensions
        and semantic.query_cache_hits + semantic.query_cache_misses == 20
        and hybrid_reuse_complete
    )
    return HeldoutQueryCacheReport(
        artifact=BaselineResolvedFile(
            path="query_embeddings.json",
            sha256=sha256_file(cache_path, "text_lf"),
            hash_mode="text_lf",
        ),
        provider=cache.provider,
        model=cache.model,
        query_representation=cache.query_representation,
        embedding_dim=cache.embedding_dim,
        entry_count=len(cache.entries),
        source_path=cache.source_path,
        source_sha256=cache.source_sha256,
        semantic_hits=semantic.query_cache_hits,
        semantic_misses=semantic.query_cache_misses,
        hybrid_hits=hybrid.query_cache_hits,
        hybrid_misses=hybrid.query_cache_misses,
        independent=independent,
        hybrid_reuse_complete=hybrid_reuse_complete,
        valid=valid,
    )


def _observational_slices(
    cases: Sequence[RetrievalEvalCase],
    configurations: Sequence[FragmentRunArtifact],
) -> HeldoutObservationalSlices:
    question_types = Counter(_metadata_label(case, "question_type") for case in cases)
    difficulties = Counter(_metadata_label(case, "difficulty") for case in cases)
    sources = Counter(case.evidence_spans[0].source_path for case in cases)
    pages = Counter(
        str(case.evidence_spans[0].page_start)
        if case.evidence_spans[0].page_start is not None
        else "unknown"
        for case in cases
    )
    ranks: dict[FragmentMode, dict[str, int]] = {}
    for run in configurations:
        counts = Counter(
            (
                f"rank-{min(case.parent_relevant_ranks)}"
                if case.parent_relevant_ranks
                else "miss"
            )
            for case in run.cases
        )
        ranks[run.retrieval_mode] = dict(sorted(counts.items()))
    return HeldoutObservationalSlices(
        question_type_counts=dict(sorted(question_types.items())),
        difficulty_counts=dict(sorted(difficulties.items())),
        source_counts=dict(sorted(sources.items())),
        evidence_page_counts=dict(sorted(pages.items())),
        relevant_rank_counts=ranks,
    )


def _metadata_label(case: RetrievalEvalCase, key: str) -> str:
    value = case.metadata.get(key)
    return value if isinstance(value, str) and value else "unknown"


def _ranking_artifact_valid(
    artifact: ScreenRunArtifact,
    *,
    manifest: FragmentPackingConfirmationManifest,
    mode: FragmentMode,
    case_ids: set[str],
) -> bool:
    evaluation = artifact.evaluation
    return (
        artifact.benchmark == manifest.name
        and artifact.variant_id == manifest.variant.id
        and artifact.workspace_snapshot_id
        == manifest.development.workspace_snapshot_id
        and artifact.cache_reuse_valid
        and evaluation.retrieval_mode == mode
        and evaluation.limit == 5
        and evaluation.case_count == 20
        and {result.id for result in evaluation.results} == case_ids
        and all(
            len(result.top_results) == 5
            and len(result.actual_chunk_ids) == 5
            for result in evaluation.results
        )
    )


def _ranking_reference(
    artifact: ScreenRunArtifact,
    *,
    mode: FragmentMode,
    ranking_path: Path,
    checkpoint_path: Path,
    root: Path,
) -> HeldoutRankingReference:
    return HeldoutRankingReference(
        retrieval_mode=mode,
        artifact=BaselineResolvedFile(
            path=_display_path(ranking_path, root),
            sha256=sha256_file(ranking_path, "text_lf"),
            hash_mode="text_lf",
        ),
        checkpoint=BaselineResolvedFile(
            path=_display_path(checkpoint_path, root),
            sha256=sha256_file(checkpoint_path, "text_lf"),
            hash_mode="text_lf",
        ),
        result_fingerprint_sha256=result_fingerprint(
            artifact.evaluation.results
        ),
        parent_hits=sum(
            bool(result.relevant_result_ranks)
            for result in artifact.evaluation.results
        ),
        query_cache_hits=artifact.query_cache_hits,
        query_cache_misses=artifact.query_cache_misses,
        prepared_cache_reuse_valid=artifact.cache_reuse_valid,
    )


def _build_ranking_checkpoint(
    artifact: ScreenRunArtifact,
    *,
    manifest: FragmentPackingConfirmationManifest,
    mode: FragmentMode,
    ranking_path: Path,
    query_cache_path: Path,
) -> HeldoutRankingCheckpoint:
    return HeldoutRankingCheckpoint(
        benchmark=manifest.name,
        variant_id=manifest.variant.id,
        git_commit=artifact.git_commit,
        workspace_snapshot_id=artifact.workspace_snapshot_id,
        retrieval_mode=mode,
        ranking_artifact_sha256=sha256_file(ranking_path, "text_lf"),
        result_fingerprint_sha256=result_fingerprint(
            artifact.evaluation.results
        ),
        query_cache_sha256=sha256_file(query_cache_path, "text_lf"),
    )


def _validate_resume_ranking(
    artifact: ScreenRunArtifact,
    *,
    checkpoint: HeldoutRankingCheckpoint,
    manifest: FragmentPackingConfirmationManifest,
    mode: FragmentMode,
    git_commit: str,
    workspace_snapshot_id: str,
    ranking_path: Path,
    query_cache_path: Path,
    case_ids: set[str],
) -> None:
    ranking_sha = sha256_file(ranking_path, "text_lf")
    query_cache_sha = sha256_file(query_cache_path, "text_lf")
    fingerprint = result_fingerprint(artifact.evaluation.results)
    if (
        not _ranking_artifact_valid(
            artifact, manifest=manifest, mode=mode, case_ids=case_ids
        )
        or artifact.git_commit != git_commit
        or artifact.workspace_snapshot_id != workspace_snapshot_id
        or checkpoint.benchmark != manifest.name
        or checkpoint.variant_id != manifest.variant.id
        or checkpoint.git_commit != git_commit
        or checkpoint.workspace_snapshot_id != workspace_snapshot_id
        or checkpoint.retrieval_mode != mode
        or checkpoint.ranking_artifact_sha256 != ranking_sha
        or checkpoint.result_fingerprint_sha256 != fingerprint
        or checkpoint.query_cache_sha256 != query_cache_sha
    ):
        raise ValueError(f"E4b held-out resume ranking mismatch: {mode}")


def _validate_resume_manifest(
    output_dir: Path,
    manifest: FragmentPackingConfirmationManifest,
) -> None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"E4b held-out resume manifest missing: {path}")
    existing = FragmentPackingConfirmationManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if existing != manifest:
        raise ValueError("E4b held-out resume manifest mismatch")


def _cache_state(cache: PreparedStateCache) -> BaselineCacheState:
    stats = cache.stats()
    return BaselineCacheState(
        snapshot_id=stats.snapshot_id,
        chunk_loads=stats.chunk_loads,
        vector_loads=stats.vector_loads,
        warm_hits=stats.warm_hits,
        invalidations=stats.invalidations,
    )


def _cache_valid(cache: BaselineCacheState) -> bool:
    return (
        cache.chunk_loads == 1
        and cache.vector_loads == 1
        and cache.invalidations == 0
    )


def _mode_case_ids(
    configurations: Sequence[FragmentRunArtifact],
    field: Literal[
        "unscorable_parent_hit_case_ids",
        "oracle_lost_hit_case_ids",
        "selector_lost_hit_case_ids",
        "candidate_representation_violations",
        "traceability_violations",
        "budget_violations",
        "mapping_violations",
    ],
) -> list[str]:
    return sorted(
        f"{run.retrieval_mode}:{case_id}"
        for run in configurations
        for case_id in getattr(run.metrics, field)
    )


def _format_optional(value: float | None) -> str:
    return "none" if value is None else f"{value:.4f}"


def _validate_file_spec(spec: BaselineFileSpec, root: Path) -> Path:
    path = (root / spec.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"E4b held-out input is invalid: {spec.path}")
    if sha256_file(path, spec.hash_mode) != spec.sha256:
        raise ValueError(f"E4b held-out input hash mismatch: {spec.path}")
    return path


def _load_cases(path: Path) -> list[RetrievalEvalCase]:
    from ragent_forge.app.services.evaluation.cases import load_cases

    return load_cases(path)


def _query_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _print_summary(
    report: FragmentPackingConfirmationReport,
    output_dir: Path,
) -> None:
    print(f"E4b held-out confirmation: {report.benchmark}")
    print(
        "Query cache: "
        f"semantic={report.query_cache.semantic_hits}h/"
        f"{report.query_cache.semantic_misses}m "
        f"hybrid={report.query_cache.hybrid_hits}h/"
        f"{report.query_cache.hybrid_misses}m"
    )
    for run in report.configurations:
        metrics = run.metrics
        print(
            f"{run.retrieval_mode}@5: parent={metrics.parent_hits} "
            f"selector-losses={len(metrics.selector_lost_hit_case_ids)} "
            f"coverage={_format_optional(metrics.average_reachable_evidence_coverage)} "
            f"efficiency={_format_optional(metrics.average_oracle_efficiency)} "
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
    print(f"Confirmed: {report.confirmed}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen E4b on the independent 20-case held-out dataset."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
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
        git_state = collect_git_state(
            root,
            ignored_untracked_roots=([args.output_dir] if args.resume else ()),
        )
        if git_state.dirty and not args.allow_dirty:
            raise ValueError(
                "E4b held-out confirmation requires a clean Git tree; use "
                "--allow-dirty only for a non-confirmable diagnostic run"
            )
        report = run_fragment_confirmation(
            load_manifest(args.manifest),
            manifest_path=args.manifest,
            repository_root=root,
            workspace=LocalWorkspace(args.workspace),
            output_dir=args.output_dir,
            git_state=git_state,
            runtime_environment=collect_runtime_environment(),
            resume=args.resume,
            progress=lambda message: print(message, flush=True),
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"E4b held-out confirmation failed: {exc}", file=sys.stderr)
        return 1
    _print_summary(report, Path(args.output_dir).resolve())
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
