from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

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
    chunk_content_fingerprint,
    index_input_fingerprint,
)
from ragent_forge.app.ports import EmbeddingServicePort
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
    BaselineTrialArtifact,
    BaselineWorkloadSpec,
    BaselineWorkspaceState,
    GitCommit,
    RetrievalBaselineManifest,
    RetrievalBaselineReport,
    compute_cutoff_metrics,
    result_fingerprint,
)
from ragent_forge.app.services.evaluation.contracts import (
    FailureType,
    RetrievalEvalCaseResult,
    RetrievalEvalReport,
)
from ragent_forge.app.services.evaluation.metrics import round_metric
from ragent_forge.app.services.evaluation.runner import RetrievalEvalService
from ragent_forge.app.services.evaluation.screening import (
    ScreenQueryCacheSummary,
    ScreenRunArtifact,
    ScreenWorkspaceFingerprints,
)
from ragent_forge.app.services.prepared_retrieval import PreparedStateCache
from ragent_forge.app.services.search_service import tokenize
from ragent_forge.app.services.vector_index_service import VectorIndexService
from ragent_forge.composition import (
    build_embedding_service,
    build_retrieval_runtime,
)
from ragent_forge.core.retrieval.context_selection import (
    ContextSelectionPolicy,
    select_ranked_prefix_with_token_budget,
)
from ragent_forge.core.retrieval.representations import (
    EmbeddingRepresentation,
    QueryEmbeddingRepresentation,
    build_query_embedding_text,
)
from ragent_forge.infrastructure.local_workspace import LocalWorkspace
from ragent_forge.infrastructure.storage import atomic_write_text

DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "direction_confirmation_manifest_e4a.json"
)

DirectionMode = Literal["semantic", "hybrid"]
DirectionTransition = Literal["retained", "gained", "lost", "unchanged_miss"]
DirectionGateName = Literal[
    "semantic_hit_direction",
    "hybrid_hit_nonnegative",
    "no_new_missed_source",
    "semantic_context_hits_retained",
    "hybrid_context_hits_retained",
    "hybrid_context_tokens",
    "context_selection_invariants",
]


class DirectionParentRunSpec(BaselineFileSpec):
    retrieval_mode: DirectionMode
    limit: Literal[5] = 5
    repetition: Literal[1, 2, 3]
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DirectionParentBaselineSpec(BaselineFileSpec):
    required_repetitions: Literal[3] = 3
    runs: list[DirectionParentRunSpec] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def _fixed_parent_runs(self) -> Self:
        keys = [(run.retrieval_mode, run.repetition) for run in self.runs]
        expected = {
            (mode, repetition)
            for mode in ("semantic", "hybrid")
            for repetition in (1, 2, 3)
        }
        if set(keys) != expected or len(keys) != len(set(keys)):
            raise ValueError("direction parent requires six unique Top-5 runs")
        self.runs = sorted(
            self.runs,
            key=lambda run: (
                0 if run.retrieval_mode == "semantic" else 1,
                run.repetition,
            ),
        )
        return self


class DirectionQueryCacheSeedSpec(BaselineFileSpec):
    entry_count: Literal[16] = 16
    query_representation: Literal["instructed_query_v1"] = "instructed_query_v1"


class DirectionCandidateSpec(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    description: str = Field(min_length=1)
    document_embedding_representation: EmbeddingRepresentation
    query_embedding_representation: QueryEmbeddingRepresentation
    workspace_build_git_commit: GitCommit
    expected_chunk_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_index_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_policy: ContextSelectionPolicy
    candidate_limit: Literal[5] = 5
    max_context_tokens: Literal[768] = 768
    characters_per_token: Literal[4] = 4

    @model_validator(mode="after")
    def _fixed_e3b_e4a_contract(self) -> Self:
        if self.document_embedding_representation != "cleaned_pdf_formula_text_v1":
            raise ValueError("direction confirmation requires E3b document input")
        if self.query_embedding_representation != "instructed_query_v1":
            raise ValueError("direction confirmation requires instructed queries")
        if self.selection_policy != "ranked_prefix_token_budget_v1":
            raise ValueError("direction confirmation requires E4a selection")
        return self

    @property
    def max_context_chars(self) -> int:
        return self.max_context_tokens * self.characters_per_token


class DirectionPromotionSpec(BaseModel):
    min_semantic_hit_rate_delta: float = Field(ge=0, le=1)
    min_hybrid_hit_rate_delta: float = Field(ge=0, le=1)
    max_new_missed_source_results: Literal[0] = 0
    max_semantic_context_hit_losses: Literal[0] = 0
    max_hybrid_context_hit_losses: Literal[0] = 0
    max_hybrid_context_token_ratio: float = Field(gt=0)
    min_selected_chunks_per_case: int = Field(gt=0)


class DirectionConfirmationManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parent_baseline: DirectionParentBaselineSpec
    dataset: BaselineDatasetSpec
    corpus: BaselineCorpusSpec
    ingest: BaselineIngestSpec
    embedding: BaselineEmbeddingSpec
    query_cache_seed: DirectionQueryCacheSeedSpec
    candidate: DirectionCandidateSpec
    promotion: DirectionPromotionSpec


class DirectionResolvedParentRun(BaseModel):
    retrieval_mode: DirectionMode
    repetition: int = Field(ge=1, le=3)
    file: BaselineResolvedFile
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DirectionResolvedParent(BaseModel):
    summary: BaselineResolvedFile
    git_commit: GitCommit
    workspace_snapshot_id: str
    runs: list[DirectionResolvedParentRun] = Field(min_length=6, max_length=6)


class DirectionParentCaseOutcome(BaseModel):
    hit_count: int = Field(ge=0, le=3)
    ranks: list[int | None] = Field(min_length=3, max_length=3)
    failure_types: list[FailureType | None] = Field(min_length=3, max_length=3)


class DirectionSelectionCaseResult(BaseModel):
    case_id: str
    parent_ranking_hit: bool
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


class DirectionSelectionMetrics(BaseModel):
    case_count: int = Field(gt=0)
    parent_ranking_hits: int = Field(ge=0)
    selected_hits: int = Field(ge=0)
    lost_hit_case_ids: list[str]
    average_selected_chunks: float = Field(ge=0)
    average_selected_context_chars: float = Field(ge=0)
    average_estimated_context_tokens: float = Field(ge=0)
    maximum_estimated_context_tokens: int = Field(ge=0)
    below_minimum_case_ids: list[str]
    over_budget_case_ids: list[str]
    invalid_prefix_case_ids: list[str]
    incomplete_mapping_case_ids: list[str]


class DirectionSelectionRunArtifact(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    candidate_id: str
    git_commit: GitCommit
    workspace_snapshot_id: str
    retrieval_mode: DirectionMode
    limit: Literal[5] = 5
    parent_ranking_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: DirectionSelectionMetrics
    cases: list[DirectionSelectionCaseResult] = Field(min_length=50, max_length=50)


class DirectionRankingCheckpoint(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    candidate_id: str
    git_commit: GitCommit
    workspace_snapshot_id: str
    retrieval_mode: DirectionMode
    ranking_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DirectionCaseComparison(BaseModel):
    case_id: str
    parent: DirectionParentCaseOutcome
    candidate_hit: bool
    candidate_rank: int | None = None
    candidate_failure_type: FailureType | None = None
    transition: DirectionTransition
    context_hit_retained: bool


class DirectionConfigurationReport(BaseModel):
    retrieval_mode: DirectionMode
    limit: Literal[5] = 5
    ranking_artifact_path: str
    ranking_checkpoint_path: str
    selection_artifact_path: str
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_metrics: BaselineCutoffMetricDistribution
    candidate_ranking_metrics: BaselineCutoffMetrics
    hit_rate_delta: float
    selection_metrics: DirectionSelectionMetrics
    query_cache_hits: int = Field(ge=0)
    query_cache_misses: int = Field(ge=0)
    cache_reuse_valid: bool
    cases: list[DirectionCaseComparison] = Field(min_length=50, max_length=50)


class DirectionGateResult(BaseModel):
    name: DirectionGateName
    passed: bool
    observed: str
    requirement: str
    case_ids: list[str] = Field(default_factory=list)


class DirectionConfirmationReport(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    description: str
    measured_at: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git: BaselineGitState
    runtime: BaselineRuntimeEnvironment
    parent_baseline: DirectionResolvedParent
    dataset: BaselineResolvedDataset
    corpus: list[BaselineResolvedFile]
    workspace: BaselineWorkspaceState
    workspace_fingerprints: ScreenWorkspaceFingerprints
    candidate: DirectionCandidateSpec
    query_cache: ScreenQueryCacheSummary
    configurations: list[DirectionConfigurationReport] = Field(
        min_length=2,
        max_length=2,
    )
    gates: list[DirectionGateResult] = Field(min_length=7, max_length=7)
    valid: bool
    confirmed: bool


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> DirectionConfirmationManifest:
    return DirectionConfirmationManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def run_direction_confirmation(
    manifest: DirectionConfirmationManifest,
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
) -> DirectionConfirmationReport:
    root = Path(repository_root).resolve()
    source_manifest_path = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and not resume:
        raise FileExistsError(
            f"Direction confirmation output directory already exists: {destination}"
        )
    if resume and not destination.is_dir():
        raise FileNotFoundError(
            f"Direction confirmation resume directory does not exist: {destination}"
        )

    validated = _validate_inputs(manifest, root, workspace)
    parent, parent_reports, parent_metrics = _load_parent_baseline(manifest, root)
    fingerprints = _workspace_fingerprints(
        workspace,
        [file.path for file in manifest.corpus.files],
    )
    if fingerprints.chunk_content_sha256 != (
        manifest.candidate.expected_chunk_content_sha256
    ):
        raise ValueError("Direction workspace chunk-content fingerprint mismatch")
    if fingerprints.index_input_sha256 != (
        manifest.candidate.expected_index_input_sha256
    ):
        raise ValueError("Direction workspace index-input fingerprint mismatch")

    if resume:
        _validate_resume_manifest(destination, manifest)
    else:
        destination.mkdir(parents=True)
        (destination / "ranking-runs").mkdir()
        (destination / "selection-runs").mkdir()
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
        query_representation=manifest.candidate.query_embedding_representation,
        embedding_dim=manifest.embedding.dimensions,
        source_path=(None if resume else root / manifest.query_cache_seed.path),
        resume=resume,
    )
    _validate_query_cache_lineage(query_cache.cache, manifest, root)
    prepared_cache = PreparedStateCache(tokenize)
    eval_service = RetrievalEvalService()
    configurations: list[DirectionConfigurationReport] = []
    for mode in ("semantic", "hybrid"):
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
            checkpoint = DirectionRankingCheckpoint.model_validate_json(
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
            )
            if progress is not None:
                progress(f"Reused {mode}@5 ranking")
        else:
            if progress is not None:
                progress(f"Starting {mode}@5 ranking over 50 cases")
            before = query_cache.stats()
            runtime = build_retrieval_runtime(
                workspace,
                mode,
                limit=5,
                config=validated.config,
                prepared_state_cache=prepared_cache,
                embedding_service=query_cache,
            )
            evaluation = eval_service.evaluate(
                cases=validated.cases,
                search_service=runtime.retrieval_engine,
                limit=5,
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
                variant_id=manifest.candidate.id,
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
                progress(
                    f"Completed {mode}@5 ranking: "
                    f"hit={compute_cutoff_metrics(evaluation.results, 5).hit_rate:.4f}"
                )

        selection_relative = Path("selection-runs") / f"{mode}-k5.json"
        selection = build_selection_artifact(
            manifest,
            ranking_artifact=ranking_artifact,
        )
        atomic_write_text(
            destination / selection_relative,
            selection.model_dump_json(indent=2) + "\n",
        )
        configurations.append(
            build_direction_configuration(
                mode=mode,
                ranking_artifact=ranking_artifact,
                selection=selection,
                parent_reports=parent_reports[mode],
                parent_metrics=parent_metrics[mode],
                ranking_artifact_path=ranking_relative.as_posix(),
                ranking_checkpoint_path=checkpoint_relative.as_posix(),
                selection_artifact_path=selection_relative.as_posix(),
            )
        )

    gates = evaluate_direction_gates(manifest, configurations)
    cache_data = QueryEmbeddingCacheFile.model_validate_json(
        query_cache_path.read_text(encoding="utf-8")
    )
    required_query_keys = {
        _query_key(
            build_query_embedding_text(
                case.query,
                manifest.candidate.query_embedding_representation,
            )
        )
        for case in validated.cases
    }
    cache_complete = set(cache_data.entries) >= required_query_keys
    report = DirectionConfirmationReport(
        benchmark=manifest.name,
        description=manifest.description,
        measured_at=datetime.now(UTC).isoformat(),
        manifest_path=_display_path(source_manifest_path, root),
        manifest_sha256=sha256_file(source_manifest_path, "text_lf"),
        git=git_state,
        runtime=runtime_environment,
        parent_baseline=parent,
        dataset=validated.dataset,
        corpus=validated.corpus,
        workspace=validated.workspace,
        workspace_fingerprints=fingerprints,
        candidate=manifest.candidate,
        query_cache=ScreenQueryCacheSummary(
            artifact_path="query_embeddings.json",
            source_path=cache_data.source_path,
            source_sha256=cache_data.source_sha256,
            sha256=sha256_file(query_cache_path, "text_lf"),
            provider=cache_data.provider,
            model=cache_data.model,
            query_representation=cache_data.query_representation,
            embedding_dim=cache_data.embedding_dim,
            entry_count=len(cache_data.entries),
            hits=sum(item.query_cache_hits for item in configurations),
            misses=sum(item.query_cache_misses for item in configurations),
        ),
        configurations=configurations,
        gates=gates,
        valid=(
            validated.checks.passed
            and not git_state.dirty
            and cache_complete
            and len(cache_data.entries) == manifest.dataset.case_count
            and all(item.cache_reuse_valid for item in configurations)
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


def build_selection_artifact(
    manifest: DirectionConfirmationManifest,
    *,
    ranking_artifact: ScreenRunArtifact,
) -> DirectionSelectionRunArtifact:
    cases = [
        _select_case(result, manifest.candidate)
        for result in ranking_artifact.evaluation.results
    ]
    lost = [case.case_id for case in cases if not case.hit_retained]
    metrics = DirectionSelectionMetrics(
        case_count=len(cases),
        parent_ranking_hits=sum(case.parent_ranking_hit for case in cases),
        selected_hits=sum(case.selected_hit for case in cases),
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
        below_minimum_case_ids=[
            case.case_id
            for case in cases
            if case.selected_count
            < manifest.promotion.min_selected_chunks_per_case
        ],
        over_budget_case_ids=[
            case.case_id for case in cases if not case.budget_respected
        ],
        invalid_prefix_case_ids=[
            case.case_id for case in cases if not case.ranked_prefix_preserved
        ],
        incomplete_mapping_case_ids=[
            case.case_id for case in cases if case.mapping_coverage != 1.0
        ],
    )
    mode_value = ranking_artifact.evaluation.retrieval_mode
    if mode_value == "semantic":
        resolved_mode: DirectionMode = "semantic"
    elif mode_value == "hybrid":
        resolved_mode = "hybrid"
    else:
        raise ValueError("Direction selection requires semantic or hybrid ranking")
    return DirectionSelectionRunArtifact(
        benchmark=manifest.name,
        candidate_id=manifest.candidate.id,
        git_commit=ranking_artifact.git_commit,
        workspace_snapshot_id=ranking_artifact.workspace_snapshot_id,
        retrieval_mode=resolved_mode,
        parent_ranking_fingerprint_sha256=result_fingerprint(
            ranking_artifact.evaluation.results
        ),
        selection_fingerprint_sha256=_selection_fingerprint(cases),
        metrics=metrics,
        cases=cases,
    )


def build_direction_configuration(
    *,
    mode: DirectionMode,
    ranking_artifact: ScreenRunArtifact,
    selection: DirectionSelectionRunArtifact,
    parent_reports: Sequence[RetrievalEvalReport],
    parent_metrics: BaselineCutoffMetricDistribution,
    ranking_artifact_path: str,
    ranking_checkpoint_path: str,
    selection_artifact_path: str,
) -> DirectionConfigurationReport:
    if len(parent_reports) != 3:
        raise ValueError("direction confirmation requires three parent trials")
    candidate_results = ranking_artifact.evaluation.results
    candidate_by_id = {result.id: result for result in candidate_results}
    selection_by_id = {case.case_id: case for case in selection.cases}
    parent_by_trial = [
        {result.id: result for result in report.results}
        for report in parent_reports
    ]
    case_ids = [result.id for result in candidate_results]
    if any(set(trial) != set(case_ids) for trial in parent_by_trial):
        raise ValueError("parent and candidate case sets differ")
    comparisons: list[DirectionCaseComparison] = []
    for case_id in case_ids:
        candidate = candidate_by_id[case_id]
        parent_results = [trial[case_id] for trial in parent_by_trial]
        parent_hit_count = sum(_result_hit(result) for result in parent_results)
        candidate_hit = _result_hit(candidate)
        parent_consensus_hit = parent_hit_count >= 2
        transition: DirectionTransition
        if parent_consensus_hit and candidate_hit:
            transition = "retained"
        elif not parent_consensus_hit and candidate_hit:
            transition = "gained"
        elif parent_consensus_hit and not candidate_hit:
            transition = "lost"
        else:
            transition = "unchanged_miss"
        comparisons.append(
            DirectionCaseComparison(
                case_id=case_id,
                parent=DirectionParentCaseOutcome(
                    hit_count=parent_hit_count,
                    ranks=[result.rank for result in parent_results],
                    failure_types=[
                        result.failure_type for result in parent_results
                    ],
                ),
                candidate_hit=candidate_hit,
                candidate_rank=candidate.rank,
                candidate_failure_type=candidate.failure_type,
                transition=transition,
                context_hit_retained=selection_by_id[case_id].hit_retained,
            )
        )
    candidate_metrics = compute_cutoff_metrics(candidate_results, 5)
    return DirectionConfigurationReport(
        retrieval_mode=mode,
        ranking_artifact_path=ranking_artifact_path,
        ranking_checkpoint_path=ranking_checkpoint_path,
        selection_artifact_path=selection_artifact_path,
        result_fingerprint_sha256=result_fingerprint(candidate_results),
        parent_metrics=parent_metrics,
        candidate_ranking_metrics=candidate_metrics,
        hit_rate_delta=round_metric(
            candidate_metrics.hit_rate - parent_metrics.hit_rate.average
        ),
        selection_metrics=selection.metrics,
        query_cache_hits=ranking_artifact.query_cache_hits,
        query_cache_misses=ranking_artifact.query_cache_misses,
        cache_reuse_valid=ranking_artifact.cache_reuse_valid,
        cases=comparisons,
    )


def evaluate_direction_gates(
    manifest: DirectionConfirmationManifest,
    configurations: Sequence[DirectionConfigurationReport],
) -> list[DirectionGateResult]:
    by_mode = {item.retrieval_mode: item for item in configurations}
    if set(by_mode) != {"semantic", "hybrid"}:
        raise ValueError("direction confirmation requires semantic and hybrid")
    semantic = by_mode["semantic"]
    hybrid = by_mode["hybrid"]
    promotion = manifest.promotion
    new_missed_source = sorted(
        f"{configuration.retrieval_mode}:{case.case_id}"
        for configuration in configurations
        for case in configuration.cases
        if case.candidate_failure_type == "missed_source"
        and "missed_source" not in case.parent.failure_types
    )
    semantic_losses = semantic.selection_metrics.lost_hit_case_ids
    hybrid_losses = hybrid.selection_metrics.lost_hit_case_ids
    hybrid_token_ratio = round_metric(
        hybrid.selection_metrics.average_estimated_context_tokens
        / hybrid.parent_metrics.avg_selected_context_tokens.average
    )
    invariant_violations = _selection_invariant_violations(
        manifest,
        configurations,
    )
    return [
        DirectionGateResult(
            name="semantic_hit_direction",
            passed=(
                semantic.hit_rate_delta
                >= promotion.min_semantic_hit_rate_delta
            ),
            observed=f"delta={semantic.hit_rate_delta:.4f}",
            requirement=f"delta>={promotion.min_semantic_hit_rate_delta:.4f}",
        ),
        DirectionGateResult(
            name="hybrid_hit_nonnegative",
            passed=(
                hybrid.hit_rate_delta >= promotion.min_hybrid_hit_rate_delta
            ),
            observed=f"delta={hybrid.hit_rate_delta:.4f}",
            requirement=f"delta>={promotion.min_hybrid_hit_rate_delta:.4f}",
        ),
        DirectionGateResult(
            name="no_new_missed_source",
            passed=(
                len(new_missed_source)
                <= promotion.max_new_missed_source_results
            ),
            observed=f"new_results={len(new_missed_source)}",
            requirement=(
                "new_results<="
                f"{promotion.max_new_missed_source_results}"
            ),
            case_ids=new_missed_source,
        ),
        DirectionGateResult(
            name="semantic_context_hits_retained",
            passed=(
                len(semantic_losses)
                <= promotion.max_semantic_context_hit_losses
            ),
            observed=f"losses={len(semantic_losses)}",
            requirement=(
                f"losses<={promotion.max_semantic_context_hit_losses}"
            ),
            case_ids=semantic_losses,
        ),
        DirectionGateResult(
            name="hybrid_context_hits_retained",
            passed=(
                len(hybrid_losses)
                <= promotion.max_hybrid_context_hit_losses
            ),
            observed=f"losses={len(hybrid_losses)}",
            requirement=f"losses<={promotion.max_hybrid_context_hit_losses}",
            case_ids=hybrid_losses,
        ),
        DirectionGateResult(
            name="hybrid_context_tokens",
            passed=(
                hybrid_token_ratio
                <= promotion.max_hybrid_context_token_ratio
            ),
            observed=f"ratio={hybrid_token_ratio:.4f}",
            requirement=(
                "ratio<="
                f"{promotion.max_hybrid_context_token_ratio:.4f}"
            ),
        ),
        DirectionGateResult(
            name="context_selection_invariants",
            passed=(len(invariant_violations) == 0),
            observed=f"violations={len(invariant_violations)}",
            requirement="violations=0",
            case_ids=invariant_violations,
        ),
    ]


def _validate_inputs(
    manifest: DirectionConfirmationManifest,
    repository_root: Path,
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
        repository_root,
        workspace,
        manifest.candidate.workspace_build_git_commit,
    )
    seed_path = _resolve_repository_path(
        repository_root,
        manifest.query_cache_seed.path,
    )
    if (
        sha256_file(seed_path, manifest.query_cache_seed.hash_mode)
        != manifest.query_cache_seed.sha256
    ):
        raise ValueError("Direction query-cache seed hash mismatch")
    seed = QueryEmbeddingCacheFile.model_validate_json(
        seed_path.read_text(encoding="utf-8")
    )
    if (
        len(seed.entries) != manifest.query_cache_seed.entry_count
        or seed.query_representation
        != manifest.query_cache_seed.query_representation
        or seed.provider != manifest.embedding.provider
        or seed.model != manifest.embedding.model
        or seed.embedding_dim != manifest.embedding.dimensions
    ):
        raise ValueError("Direction query-cache seed configuration mismatch")
    required_query_keys = {
        _query_key(
            build_query_embedding_text(
                case.query,
                manifest.candidate.query_embedding_representation,
            )
        )
        for case in validated.cases
    }
    if len(required_query_keys) != manifest.dataset.case_count:
        raise ValueError("Direction dataset queries are not unique")
    if not set(seed.entries).issubset(required_query_keys):
        raise ValueError("Direction query-cache seed contains an unrelated query")
    return validated


def _load_parent_baseline(
    manifest: DirectionConfirmationManifest,
    repository_root: Path,
) -> tuple[
    DirectionResolvedParent,
    dict[DirectionMode, list[RetrievalEvalReport]],
    dict[DirectionMode, BaselineCutoffMetricDistribution],
]:
    spec = manifest.parent_baseline
    summary_path = _resolve_repository_path(repository_root, spec.path)
    digest = sha256_file(summary_path, spec.hash_mode)
    if digest != spec.sha256:
        raise ValueError("Direction parent baseline summary hash mismatch")
    summary = RetrievalBaselineReport.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    if not summary.passed:
        raise ValueError("Direction parent baseline did not pass")
    if summary.git.dirty or not summary.input_checks.passed:
        raise ValueError("Direction parent baseline provenance is invalid")
    parent_configurations = {
        mode: [
            item
            for item in summary.configurations
            if item.retrieval_mode == mode and item.limit == 5
        ]
        for mode in ("semantic", "hybrid")
    }
    if any(len(items) != 1 for items in parent_configurations.values()):
        raise ValueError(
            "Direction parent baseline requires one Semantic@5 and one Hybrid@5 "
            "configuration"
        )
    reports: dict[DirectionMode, list[RetrievalEvalReport]] = {
        "semantic": [],
        "hybrid": [],
    }
    resolved_runs: list[DirectionResolvedParentRun] = []
    for run_spec in spec.runs:
        path = _resolve_repository_path(repository_root, run_spec.path)
        run_digest = sha256_file(path, run_spec.hash_mode)
        if run_digest != run_spec.sha256:
            raise ValueError(f"Direction parent run hash mismatch: {run_spec.path}")
        artifact = BaselineTrialArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        configuration = parent_configurations[run_spec.retrieval_mode][0]
        matching_trials = [
            trial
            for trial in configuration.trials
            if trial.repetition == run_spec.repetition
        ]
        if len(matching_trials) != 1:
            raise ValueError("Direction parent summary trial linkage mismatch")
        summary_trial = matching_trials[0]
        summary_artifact_path = (
            summary_path.parent / summary_trial.artifact_path
        ).resolve()
        if (
            summary_artifact_path != path
            or summary_trial.result_fingerprint_sha256
            != run_spec.result_fingerprint_sha256
            or artifact.benchmark != summary.benchmark
            or artifact.git_commit not in summary.trial_git_commits
            or artifact.workspace_build_git_commit
            != summary.workspace.build_git_commit
            or artifact.workspace_snapshot_id != summary.workspace.snapshot_id
            or artifact.evaluation.retrieval_mode != run_spec.retrieval_mode
            or artifact.evaluation.limit != 5
            or artifact.evaluation.case_count != manifest.dataset.case_count
            or len(artifact.evaluation.results) != manifest.dataset.case_count
            or artifact.trial.repetition != run_spec.repetition
            or artifact.trial.retrieval_mode != run_spec.retrieval_mode
            or artifact.trial.limit != 5
            or artifact.trial.result_fingerprint_sha256
            != run_spec.result_fingerprint_sha256
            or result_fingerprint(artifact.evaluation.results)
            != run_spec.result_fingerprint_sha256
        ):
            raise ValueError("Direction parent run provenance mismatch")
        reports[run_spec.retrieval_mode].append(artifact.evaluation)
        resolved_runs.append(
            DirectionResolvedParentRun(
                retrieval_mode=run_spec.retrieval_mode,
                repetition=run_spec.repetition,
                file=BaselineResolvedFile(
                    path=_display_path(path, repository_root),
                    sha256=run_digest,
                    hash_mode=run_spec.hash_mode,
                ),
                result_fingerprint_sha256=run_spec.result_fingerprint_sha256,
            )
        )
    metrics: dict[DirectionMode, BaselineCutoffMetricDistribution] = {}
    for mode in ("semantic", "hybrid"):
        configuration = parent_configurations[mode][0]
        metrics[mode] = configuration.metrics_by_cutoff[5]
    return (
        DirectionResolvedParent(
            summary=BaselineResolvedFile(
                path=_display_path(summary_path, repository_root),
                sha256=digest,
                hash_mode=spec.hash_mode,
            ),
            git_commit=summary.git.commit,
            workspace_snapshot_id=summary.workspace.snapshot_id,
            runs=resolved_runs,
        ),
        reports,
        metrics,
    )


def _select_case(
    result: RetrievalEvalCaseResult,
    candidate: DirectionCandidateSpec,
) -> DirectionSelectionCaseResult:
    ranked = [_ranked_item(item) for item in result.top_results]
    ranked_ids = [item[1] for item in ranked]
    if ranked_ids != result.actual_chunk_ids:
        raise ValueError(f"Direction ranked chunk ids mismatch: {result.id}")
    if [item[0] for item in ranked] != list(range(1, len(ranked) + 1)):
        raise ValueError(f"Direction ranks are not contiguous: {result.id}")
    selected = select_ranked_prefix_with_token_budget(
        ranked,
        limit=candidate.candidate_limit,
        max_context_tokens=candidate.max_context_tokens,
        characters_per_token=candidate.characters_per_token,
        text_length=lambda item: item[2],
    )
    relevant = set(result.relevant_result_ranks)
    selected_ranks = [item[0] for item in selected]
    parent_hit = bool(relevant.intersection(range(1, 6)))
    selected_hit = bool(relevant.intersection(selected_ranks))
    selected_chars = sum(item[2] for item in selected)
    mapping_coverage = result.mapping_coverage
    if mapping_coverage is None:
        raise ValueError(f"Direction mapping coverage missing: {result.id}")
    selected_ids = [item[1] for item in selected]
    estimated_tokens = math.ceil(
        selected_chars / candidate.characters_per_token
    )
    return DirectionSelectionCaseResult(
        case_id=result.id,
        parent_ranking_hit=parent_hit,
        relevant_ranks=result.relevant_result_ranks,
        ranked_chunk_ids=ranked_ids,
        selected_chunk_ids=selected_ids,
        selected_ranks=selected_ranks,
        selected_count=len(selected),
        selected_context_chars=selected_chars,
        estimated_context_tokens=estimated_tokens,
        selected_hit=selected_hit,
        hit_retained=(not parent_hit or selected_hit),
        context_nonempty=(len(selected) > 0),
        budget_respected=(estimated_tokens <= candidate.max_context_tokens),
        ranked_prefix_preserved=(selected_ids == ranked_ids[: len(selected_ids)]),
        mapping_coverage=mapping_coverage,
    )


def _ranked_item(payload: Mapping[str, object]) -> tuple[int, str, int]:
    rank = payload.get("rank")
    chunk_id = payload.get("chunk_id")
    text_chars = payload.get("text_chars")
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank < 1
        or not isinstance(chunk_id, str)
        or not chunk_id
        or not isinstance(text_chars, int)
        or isinstance(text_chars, bool)
        or text_chars < 0
    ):
        raise ValueError("Direction parent top-result shape is invalid")
    return rank, chunk_id, text_chars


def _selection_invariant_violations(
    manifest: DirectionConfirmationManifest,
    configurations: Sequence[DirectionConfigurationReport],
) -> list[str]:
    violations: list[str] = []
    for configuration in configurations:
        mode = configuration.retrieval_mode
        metrics = configuration.selection_metrics
        violations.extend(
            f"{mode}:{case_id}:below_minimum"
            for case_id in metrics.below_minimum_case_ids
        )
        violations.extend(
            f"{mode}:{case_id}:over_budget"
            for case_id in metrics.over_budget_case_ids
        )
        violations.extend(
            f"{mode}:{case_id}:invalid_prefix"
            for case_id in metrics.invalid_prefix_case_ids
        )
        violations.extend(
            f"{mode}:{case_id}:incomplete_mapping"
            for case_id in metrics.incomplete_mapping_case_ids
        )
        if (
            metrics.maximum_estimated_context_tokens
            > manifest.candidate.max_context_tokens
        ):
            violations.append(f"{mode}:maximum_budget")
    return violations


def _selection_fingerprint(
    cases: Sequence[DirectionSelectionCaseResult],
) -> str:
    payload = [
        {
            "case_id": case.case_id,
            "selected_chunk_ids": case.selected_chunk_ids,
            "selected_context_chars": case.selected_context_chars,
            "estimated_context_tokens": case.estimated_context_tokens,
        }
        for case in cases
    ]
    return _payload_sha256(payload)


def _workspace_fingerprints(
    workspace: LocalWorkspace,
    corpus_paths: Sequence[str],
) -> ScreenWorkspaceFingerprints:
    chunks = workspace.read_chunks()
    records = VectorIndexService(workspace).read_index()
    if {str(chunk.get("chunk_id", "")) for chunk in chunks} != {
        record.chunk_id for record in records
    }:
        raise ValueError("Direction index chunk ids do not match chunks")
    return ScreenWorkspaceFingerprints(
        chunk_content_sha256=chunk_content_fingerprint(chunks, corpus_paths),
        index_input_sha256=index_input_fingerprint(records, corpus_paths),
    )


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


def _validate_query_cache_lineage(
    cache: QueryEmbeddingCacheFile,
    manifest: DirectionConfirmationManifest,
    repository_root: Path,
) -> None:
    seed_path = _resolve_repository_path(
        repository_root,
        manifest.query_cache_seed.path,
    )
    source_path = Path(cache.source_path).resolve() if cache.source_path else None
    if (
        source_path != seed_path
        or cache.source_sha256 != manifest.query_cache_seed.sha256
    ):
        raise ValueError("Direction query-cache seed lineage mismatch")
    seed = QueryEmbeddingCacheFile.model_validate_json(
        seed_path.read_text(encoding="utf-8")
    )
    if any(cache.entries.get(key) != vector for key, vector in seed.entries.items()):
        raise ValueError("Direction query-cache seed vectors changed")


def _result_hit(result: RetrievalEvalCaseResult) -> bool:
    return any(rank <= 5 for rank in result.relevant_result_ranks)


def _validate_resume_manifest(
    output_dir: Path,
    manifest: DirectionConfirmationManifest,
) -> None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Direction resume manifest missing: {path}")
    existing = DirectionConfirmationManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if existing != manifest:
        raise ValueError("Direction resume manifest mismatch")


def _validate_resume_ranking(
    artifact: ScreenRunArtifact,
    *,
    checkpoint: DirectionRankingCheckpoint,
    manifest: DirectionConfirmationManifest,
    mode: DirectionMode,
    git_commit: str,
    workspace_snapshot_id: str,
    ranking_path: Path,
    query_cache_path: Path,
) -> None:
    ranking_sha256 = sha256_file(ranking_path, "text_lf")
    query_cache_sha256 = sha256_file(query_cache_path, "text_lf")
    fingerprint = result_fingerprint(artifact.evaluation.results)
    if (
        artifact.benchmark != manifest.name
        or artifact.variant_id != manifest.candidate.id
        or artifact.git_commit != git_commit
        or artifact.workspace_snapshot_id != workspace_snapshot_id
        or artifact.evaluation.retrieval_mode != mode
        or artifact.evaluation.limit != 5
        or artifact.evaluation.case_count != manifest.dataset.case_count
        or len(artifact.evaluation.results) != manifest.dataset.case_count
        or not artifact.cache_reuse_valid
        or checkpoint.benchmark != manifest.name
        or checkpoint.candidate_id != manifest.candidate.id
        or checkpoint.git_commit != git_commit
        or checkpoint.workspace_snapshot_id != workspace_snapshot_id
        or checkpoint.retrieval_mode != mode
        or checkpoint.ranking_artifact_sha256 != ranking_sha256
        or checkpoint.result_fingerprint_sha256 != fingerprint
        or checkpoint.query_cache_sha256 != query_cache_sha256
    ):
        raise ValueError(f"Direction resume ranking mismatch: {mode}")


def _build_ranking_checkpoint(
    artifact: ScreenRunArtifact,
    *,
    manifest: DirectionConfirmationManifest,
    mode: DirectionMode,
    ranking_path: Path,
    query_cache_path: Path,
) -> DirectionRankingCheckpoint:
    return DirectionRankingCheckpoint(
        benchmark=manifest.name,
        candidate_id=manifest.candidate.id,
        git_commit=artifact.git_commit,
        workspace_snapshot_id=artifact.workspace_snapshot_id,
        retrieval_mode=mode,
        ranking_artifact_sha256=sha256_file(ranking_path, "text_lf"),
        result_fingerprint_sha256=result_fingerprint(
            artifact.evaluation.results
        ),
        query_cache_sha256=sha256_file(query_cache_path, "text_lf"),
    )


def _query_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to(repository_root):
        raise ValueError(f"Direction path escapes repository root: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"Direction input file not found: {path}")
    return path


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _print_summary(report: DirectionConfirmationReport, output_dir: Path) -> None:
    print(f"Direction confirmation: {report.benchmark}")
    print(f"Candidate: {report.candidate.id}")
    print(f"Cases: {report.dataset.case_count}")
    print(
        "Query cache: "
        f"{report.query_cache.hits} hits, {report.query_cache.misses} misses, "
        f"{report.query_cache.entry_count} entries"
    )
    for configuration in report.configurations:
        print(
            f"{configuration.retrieval_mode}@5: "
            f"parent={configuration.parent_metrics.hit_rate.average:.4f} "
            f"candidate={configuration.candidate_ranking_metrics.hit_rate:.4f} "
            f"delta={configuration.hit_rate_delta:.4f} "
            f"context-tokens="
            f"{configuration.selection_metrics.average_estimated_context_tokens:.4f}"
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
        description="Run E3b + E4a full 50-case direction confirmation."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the checked-in direction confirmation manifest.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Prepared E3b generation-layout workspace.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for direction confirmation artifacts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse completed ranking artifacts and query cache.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty-tree diagnostic run that cannot be confirmed.",
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
        git_state = collect_git_state(
            repository_root,
            ignored_untracked_roots=([args.output_dir] if args.resume else ()),
        )
        if git_state.dirty and not args.allow_dirty:
            raise ValueError(
                "Direction confirmation requires a clean Git tree; use "
                "--allow-dirty only for a non-confirmable diagnostic run"
            )
        report = run_direction_confirmation(
            load_manifest(args.manifest),
            manifest_path=args.manifest,
            repository_root=repository_root,
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
        print(f"Direction confirmation failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, Path(args.output_dir).resolve())
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
