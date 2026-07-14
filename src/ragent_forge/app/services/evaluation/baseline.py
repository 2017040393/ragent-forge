from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from ragent_forge.app.services.evaluation.contracts import (
    RetrievalEvalCaseResult,
    RetrievalEvalReport,
    RetrievalStageLatencySummary,
)
from ragent_forge.app.services.evaluation.metrics import (
    ndcg_at,
    percentile,
    round_metric,
    summarize_stage_latencies,
)
from ragent_forge.core.retrieval.types import RetrievalMode

HashMode = Literal["binary", "text_lf"]


class BaselineFileSpec(BaseModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hash_mode: HashMode = "binary"


class BaselineDatasetSpec(BaselineFileSpec):
    case_count: int = Field(ge=2)
    manifest: BaselineFileSpec


class BaselineCorpusSpec(BaseModel):
    root: str = Field(min_length=1)
    files: list[BaselineFileSpec] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def _unique_file_paths(
        cls,
        files: list[BaselineFileSpec],
    ) -> list[BaselineFileSpec]:
        paths = [item.path for item in files]
        if len(paths) != len(set(paths)):
            raise ValueError("corpus file paths must be unique")
        return files


class BaselineIngestSpec(BaseModel):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    expected_document_count: int = Field(gt=0)
    expected_chunk_count: int = Field(gt=0)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class BaselineEmbeddingSpec(BaseModel):
    provider: Literal["openai_embeddings"] = "openai_embeddings"
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)


class BaselineWorkloadSpec(BaseModel):
    retrieval_modes: list[RetrievalMode] = Field(min_length=1)
    max_limit: int = Field(gt=0)
    cutoffs: list[int] = Field(min_length=1)
    repetitions: int = Field(ge=3)
    cold_definition: Literal["first_query_of_isolated_runtime"] = (
        "first_query_of_isolated_runtime"
    )
    warm_definition: Literal["remaining_queries_of_same_runtime"] = (
        "remaining_queries_of_same_runtime"
    )

    @field_validator("retrieval_modes")
    @classmethod
    def _unique_modes(cls, modes: list[RetrievalMode]) -> list[RetrievalMode]:
        if len(modes) != len(set(modes)):
            raise ValueError("retrieval_modes must be unique")
        return modes

    @field_validator("cutoffs")
    @classmethod
    def _positive_unique_cutoffs(cls, cutoffs: list[int]) -> list[int]:
        if any(cutoff < 1 for cutoff in cutoffs):
            raise ValueError("cutoffs must be positive integers")
        if len(cutoffs) != len(set(cutoffs)):
            raise ValueError("cutoffs must be unique")
        return sorted(cutoffs)

    @model_validator(mode="after")
    def _cutoffs_fit_retrieval_limit(self) -> Self:
        if self.cutoffs[-1] > self.max_limit:
            raise ValueError("cutoffs must not exceed max_limit")
        return self


class RetrievalBaselineManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    dataset: BaselineDatasetSpec
    corpus: BaselineCorpusSpec
    ingest: BaselineIngestSpec
    embedding: BaselineEmbeddingSpec | None = None
    workload: BaselineWorkloadSpec

    @model_validator(mode="after")
    def _dense_modes_require_embedding(self) -> Self:
        dense_modes = {"semantic", "hybrid"}
        if dense_modes.intersection(self.workload.retrieval_modes) and (
            self.embedding is None
        ):
            raise ValueError("semantic or hybrid baseline modes require embedding")
        return self


class BaselineLatencySummary(BaseModel):
    sample_count: int = Field(gt=0)
    average_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


class BaselineCutoffMetrics(BaseModel):
    cutoff: int = Field(gt=0)
    hit_rate: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    ndcg: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    avg_selected_context_chars: float = Field(ge=0)
    avg_selected_context_tokens: float = Field(ge=0)


class BaselineCacheState(BaseModel):
    snapshot_id: str | None = None
    chunk_loads: int = Field(ge=0)
    vector_loads: int = Field(ge=0)
    warm_hits: int = Field(ge=0)
    invalidations: int = Field(ge=0)


class BaselineTrialReport(BaseModel):
    repetition: int = Field(gt=0)
    retrieval_mode: RetrievalMode
    retrieval_method: str = Field(min_length=1)
    max_limit: int = Field(gt=0)
    artifact_path: str = Field(min_length=1)
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics_by_cutoff: dict[int, BaselineCutoffMetrics]
    cold_start_latency_ms: float = Field(ge=0)
    warm_latency_samples_ms: list[float] = Field(min_length=1)
    warm_latency_ms: BaselineLatencySummary
    cold_stage_latency_ms: dict[str, RetrievalStageLatencySummary]
    warm_stage_latency_ms: dict[str, RetrievalStageLatencySummary]
    cache: BaselineCacheState
    cache_reuse_valid: bool


class BaselineConfigurationReport(BaseModel):
    retrieval_mode: RetrievalMode
    retrieval_method: str = Field(min_length=1)
    max_limit: int = Field(gt=0)
    quality_stable: bool
    metrics_by_cutoff: dict[int, BaselineCutoffMetrics]
    cold_start_latency_ms: BaselineLatencySummary
    warm_latency_ms: BaselineLatencySummary
    trials: list[BaselineTrialReport] = Field(min_length=1)
    passed: bool


class BaselineGitState(BaseModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    branch: str | None = None
    dirty: bool


class BaselinePackageVersions(BaseModel):
    ragent_forge: str
    pydantic: str
    httpx: str
    pdfplumber: str


class BaselineRuntimeEnvironment(BaseModel):
    python: str
    implementation: str
    platform: str
    machine: str
    processor: str
    packages: BaselinePackageVersions


class BaselineResolvedFile(BaseModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hash_mode: HashMode


class BaselineResolvedDataset(BaseModel):
    cases: BaselineResolvedFile
    manifest: BaselineResolvedFile
    case_count: int = Field(ge=2)


class BaselineIndexState(BaseModel):
    embedding_provider: str
    embedding_model: str
    embedding_dim: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    snapshot_id: str


class BaselineWorkspaceState(BaseModel):
    root: str
    layout: Literal["generation"] = "generation"
    schema_version: int = Field(gt=0)
    snapshot_id: str
    source_path: str
    document_count: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_batch_size: int | None = None
    embedding_timeout_seconds: int | None = None
    index: BaselineIndexState | None = None


class BaselineInputChecks(BaseModel):
    dataset_hash: bool
    dataset_manifest_hash: bool
    source_hashes: bool
    case_count: bool
    generation_layout: bool
    snapshot_manifest: bool
    ingest_configuration: bool
    workspace_counts: bool
    embedding_configuration: bool
    vector_index: bool

    @property
    def passed(self) -> bool:
        return all(self.model_dump().values())


class RetrievalBaselineReport(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    description: str
    measured_at: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git: BaselineGitState
    runtime: BaselineRuntimeEnvironment
    dataset: BaselineResolvedDataset
    corpus: list[BaselineResolvedFile]
    workspace: BaselineWorkspaceState
    input_checks: BaselineInputChecks
    configurations: list[BaselineConfigurationReport]
    passed: bool


class BaselineTrialArtifact(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    workspace_snapshot_id: str
    trial: BaselineTrialReport
    evaluation: RetrievalEvalReport


def build_trial_report(
    report: RetrievalEvalReport,
    *,
    repetition: int,
    cutoffs: Sequence[int],
    artifact_path: str,
    cache: BaselineCacheState,
) -> BaselineTrialReport:
    if len(report.results) < 2:
        raise ValueError("baseline trials require at least two eval cases")
    if any(cutoff > report.limit for cutoff in cutoffs):
        raise ValueError("baseline cutoffs must not exceed the report limit")

    cold_result = report.results[0]
    warm_results = report.results[1:]
    warm_samples = [result.retrieval_latency_ms for result in warm_results]
    expected_vector_loads = 1 if report.retrieval_mode in {"semantic", "hybrid"} else 0
    cache_reuse_valid = (
        cache.chunk_loads == 1
        and cache.vector_loads == expected_vector_loads
        and cache.invalidations == 0
        and cache.warm_hits >= len(warm_results)
    )
    return BaselineTrialReport(
        repetition=repetition,
        retrieval_mode=report.retrieval_mode,
        retrieval_method=report.retrieval_method,
        max_limit=report.limit,
        artifact_path=artifact_path,
        result_fingerprint_sha256=result_fingerprint(report.results),
        metrics_by_cutoff={
            cutoff: compute_cutoff_metrics(report.results, cutoff)
            for cutoff in cutoffs
        },
        cold_start_latency_ms=cold_result.retrieval_latency_ms,
        warm_latency_samples_ms=warm_samples,
        warm_latency_ms=summarize_latency(warm_samples),
        cold_stage_latency_ms=summarize_stage_latencies([cold_result]),
        warm_stage_latency_ms=summarize_stage_latencies(warm_results),
        cache=cache,
        cache_reuse_valid=cache_reuse_valid,
    )


def build_configuration_report(
    trials: Sequence[BaselineTrialReport],
) -> BaselineConfigurationReport:
    if not trials:
        raise ValueError("baseline configuration requires at least one trial")
    first = trials[0]
    if any(
        trial.retrieval_mode != first.retrieval_mode
        or trial.retrieval_method != first.retrieval_method
        or trial.max_limit != first.max_limit
        for trial in trials
    ):
        raise ValueError("baseline configuration trials must use the same retrieval")

    quality_stable = (
        len({trial.result_fingerprint_sha256 for trial in trials}) == 1
        and all(trial.metrics_by_cutoff == first.metrics_by_cutoff for trial in trials)
    )
    warm_samples = [
        latency
        for trial in trials
        for latency in trial.warm_latency_samples_ms
    ]
    return BaselineConfigurationReport(
        retrieval_mode=first.retrieval_mode,
        retrieval_method=first.retrieval_method,
        max_limit=first.max_limit,
        quality_stable=quality_stable,
        metrics_by_cutoff=first.metrics_by_cutoff,
        cold_start_latency_ms=summarize_latency(
            [trial.cold_start_latency_ms for trial in trials]
        ),
        warm_latency_ms=summarize_latency(warm_samples),
        trials=list(trials),
        passed=quality_stable and all(trial.cache_reuse_valid for trial in trials),
    )


def compute_cutoff_metrics(
    results: Sequence[RetrievalEvalCaseResult],
    cutoff: int,
) -> BaselineCutoffMetrics:
    if cutoff < 1:
        raise ValueError("cutoff must be greater than 0")
    if not results:
        raise ValueError("cutoff metrics require at least one result")

    hits = 0
    recall_total = 0.0
    precision_total = 0.0
    ndcg_total = 0.0
    reciprocal_rank_total = 0.0
    context_chars_total = 0
    context_tokens_total = 0
    for result in results:
        relevant_ranks = [
            rank for rank in result.relevant_result_ranks if rank <= cutoff
        ]
        expected_relevant_count = result.expected_chunk_count or len(
            result.expected_source_paths
        )
        if relevant_ranks:
            hits += 1
            reciprocal_rank_total += 1 / min(relevant_ranks)
        if expected_relevant_count > 0:
            recall_total += len(relevant_ranks) / expected_relevant_count
        precision_total += len(relevant_ranks) / cutoff
        ndcg_total += ndcg_at(
            relevant_ranks,
            expected_relevant_count=expected_relevant_count,
            k=cutoff,
        )
        selected_chars = _selected_context_chars(result, cutoff)
        context_chars_total += selected_chars
        context_tokens_total += math.ceil(selected_chars / 4)

    case_count = len(results)
    return BaselineCutoffMetrics(
        cutoff=cutoff,
        hit_rate=round_metric(hits / case_count),
        recall=round_metric(recall_total / case_count),
        precision=round_metric(precision_total / case_count),
        ndcg=round_metric(ndcg_total / case_count),
        mrr=round_metric(reciprocal_rank_total / case_count),
        passed_count=hits,
        failed_count=case_count - hits,
        avg_selected_context_chars=round_metric(context_chars_total / case_count),
        avg_selected_context_tokens=round_metric(context_tokens_total / case_count),
    )


def summarize_latency(values: Sequence[float]) -> BaselineLatencySummary:
    if not values:
        raise ValueError("latency summary requires at least one sample")
    samples = [float(value) for value in values]
    return BaselineLatencySummary(
        sample_count=len(samples),
        average_ms=round_metric(sum(samples) / len(samples)),
        p50_ms=round_metric(percentile(samples, 0.5)),
        p95_ms=round_metric(percentile(samples, 0.95)),
    )


def result_fingerprint(results: Sequence[RetrievalEvalCaseResult]) -> str:
    payload = [
        {
            "id": result.id,
            "actual_chunk_ids": result.actual_chunk_ids,
        }
        for result in results
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_context_chars(
    result: RetrievalEvalCaseResult,
    cutoff: int,
) -> int:
    total = 0
    for top_result in result.top_results:
        rank = top_result.get("rank")
        text_chars = top_result.get("text_chars")
        if (
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and rank <= cutoff
            and isinstance(text_chars, int)
            and not isinstance(text_chars, bool)
            and text_chars >= 0
        ):
            total += text_chars
    return total
