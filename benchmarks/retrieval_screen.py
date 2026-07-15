from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from benchmarks.retrieval_baseline import (
    ValidatedBaselineInputs,
    collect_git_state,
    collect_runtime_environment,
    sha256_file,
    validate_inputs,
)
from ragent_forge.app.models import EmbeddingResult
from ragent_forge.app.ports import EmbeddingServicePort
from ragent_forge.app.services.evaluation.baseline import (
    BaselineCacheState,
    BaselineGitState,
    BaselineResolvedFile,
    BaselineRuntimeEnvironment,
    BaselineTrialArtifact,
    BaselineWorkloadSpec,
    RetrievalBaselineManifest,
    RetrievalBaselineReport,
)
from ragent_forge.app.services.evaluation.contracts import (
    RetrievalEvalCase,
    RetrievalEvalReport,
)
from ragent_forge.app.services.evaluation.runner import RetrievalEvalService
from ragent_forge.app.services.evaluation.screening import (
    RetrievalScreenManifest,
    RetrievalScreenReport,
    ScreenConfigurationReport,
    ScreenLimit,
    ScreenMode,
    ScreenQueryCacheSummary,
    ScreenResolvedParentBaseline,
    ScreenRunArtifact,
    ScreenWorkspaceFingerprints,
    build_screen_configuration,
    evaluate_screen_gates,
)
from ragent_forge.app.services.prepared_retrieval import PreparedStateCache
from ragent_forge.app.services.search_service import tokenize
from ragent_forge.app.services.vector_index_service import (
    VectorIndexRecord,
    VectorIndexService,
    hash_text,
)
from ragent_forge.composition import (
    build_embedding_service,
    build_retrieval_runtime,
)
from ragent_forge.core.retrieval.contracts import ChunkRecord
from ragent_forge.core.retrieval.representations import (
    QueryEmbeddingRepresentation,
    build_query_embedding_text,
)
from ragent_forge.infrastructure.local_workspace import LocalWorkspace
from ragent_forge.infrastructure.storage import atomic_write_text

DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "retrieval_screen_manifest.json"
)
ScreenKey = tuple[ScreenMode, ScreenLimit]


class QueryEmbeddingCacheFile(BaseModel):
    schema_version: Literal[1] = 1
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    query_representation: QueryEmbeddingRepresentation
    embedding_dim: int = Field(gt=0)
    source_path: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    entries: dict[str, list[float]] = Field(default_factory=dict)


@dataclass(frozen=True)
class QueryCacheStats:
    hits: int
    misses: int


class EmbeddingClientProtocol(Protocol):
    def embed_texts(self, texts: list[str]) -> EmbeddingResult: ...


class CachedQueryEmbeddingService:
    def __init__(
        self,
        delegate: EmbeddingClientProtocol,
        *,
        cache_path: str | Path,
        provider: str,
        model: str,
        query_representation: QueryEmbeddingRepresentation,
        embedding_dim: int,
        source_path: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        self.delegate = delegate
        self.provider_name = provider
        self.cache_path = Path(cache_path).resolve()
        self._hits = 0
        self._misses = 0
        if resume:
            if not self.cache_path.is_file():
                raise FileNotFoundError(
                    f"Screen query cache is missing: {self.cache_path}"
                )
            cache = self._load(self.cache_path)
        elif source_path is not None:
            resolved_source = Path(source_path).resolve()
            cache = self._load(resolved_source).model_copy(
                update={
                    "source_path": str(resolved_source),
                    "source_sha256": sha256_file(resolved_source, "text_lf"),
                }
            )
        else:
            cache = QueryEmbeddingCacheFile(
                provider=provider,
                model=model,
                query_representation=query_representation,
                embedding_dim=embedding_dim,
            )
        expected = (provider, model, query_representation, embedding_dim)
        actual = (
            cache.provider,
            cache.model,
            cache.query_representation,
            cache.embedding_dim,
        )
        if actual != expected:
            raise ValueError(
                "Screen query cache configuration mismatch: "
                f"expected {expected}, found {actual}"
            )
        self.cache = cache
        self._validate_embeddings()
        self._persist()

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        represented_texts = [
            build_query_embedding_text(text, self.cache.query_representation)
            for text in texts
        ]
        keys = [_query_key(text) for text in represented_texts]
        missing_by_key: dict[str, str] = {}
        for key, text in zip(keys, represented_texts, strict=True):
            if key in self.cache.entries:
                self._hits += 1
            else:
                self._misses += 1
                missing_by_key.setdefault(key, text)

        usage: dict[str, object] = {}
        if missing_by_key:
            missing_keys = list(missing_by_key)
            result = self.delegate.embed_texts(
                [missing_by_key[key] for key in missing_keys]
            )
            if result.provider_name != self.cache.provider:
                raise ValueError("Screen query cache provider response mismatch")
            if result.model != self.cache.model:
                raise ValueError("Screen query cache model response mismatch")
            if len(result.embeddings) != len(missing_keys):
                raise ValueError("Screen query cache embedding count mismatch")
            for key, embedding in zip(
                missing_keys,
                result.embeddings,
                strict=True,
            ):
                if len(embedding) != self.cache.embedding_dim:
                    raise ValueError("Screen query cache embedding dimension mismatch")
                self.cache.entries[key] = embedding
            usage = {str(key): value for key, value in result.usage.items()}
            self._persist()

        return EmbeddingResult(
            provider_name=self.cache.provider,
            model=self.cache.model,
            embeddings=[self.cache.entries[key] for key in keys],
            usage=usage,
            metadata={
                "screen_query_cache": True,
                "query_representation": self.cache.query_representation,
            },
        )

    def stats(self) -> QueryCacheStats:
        return QueryCacheStats(hits=self._hits, misses=self._misses)

    def _load(self, path: Path) -> QueryEmbeddingCacheFile:
        payload = path.read_text(encoding="utf-8")
        return QueryEmbeddingCacheFile.model_validate_json(payload)

    def _validate_embeddings(self) -> None:
        if any(
            len(embedding) != self.cache.embedding_dim
            for embedding in self.cache.entries.values()
        ):
            raise ValueError("Screen query cache contains an invalid embedding")

    def _persist(self) -> None:
        atomic_write_text(
            self.cache_path,
            self.cache.model_dump_json(indent=2) + "\n",
        )


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> RetrievalScreenManifest:
    payload = Path(path).read_text(encoding="utf-8")
    return RetrievalScreenManifest.model_validate_json(payload)


def run_screen(
    manifest: RetrievalScreenManifest,
    *,
    manifest_path: str | Path,
    repository_root: str | Path,
    workspace: LocalWorkspace,
    output_dir: str | Path,
    git_state: BaselineGitState,
    runtime_environment: BaselineRuntimeEnvironment,
    query_cache_source: str | Path | None = None,
    embedding_service: EmbeddingServicePort | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> RetrievalScreenReport:
    root = Path(repository_root).resolve()
    source_manifest_path = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and not resume:
        raise FileExistsError(
            f"Screen output directory already exists: {destination}"
        )
    if resume and not destination.is_dir():
        raise FileNotFoundError(
            f"Screen resume directory does not exist: {destination}"
        )

    validated = _validate_screen_inputs(manifest, root, workspace)
    parent, parent_reports = _load_parent_baseline(manifest, root)
    selected_cases = _select_cases(validated, manifest.selected_case_ids)
    fingerprints = _workspace_fingerprints(
        workspace,
        [file.path for file in manifest.corpus.files],
    )
    if fingerprints.chunk_content_sha256 != (
        manifest.variant.expected_chunk_content_sha256
    ):
        raise ValueError("Screen workspace chunk content fingerprint mismatch")
    if fingerprints.index_input_sha256 != (
        manifest.variant.expected_index_input_sha256
    ):
        raise ValueError("Screen vector index input fingerprint mismatch")

    if resume:
        _validate_resume_manifest(destination, manifest)
        _validate_resume_run_files(destination, manifest)
    else:
        destination.mkdir(parents=True)
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
        query_representation=(
            manifest.variant.query_embedding_representation
        ),
        embedding_dim=manifest.embedding.dimensions,
        source_path=query_cache_source,
        resume=resume,
    )
    prepared_cache = PreparedStateCache(tokenize)
    eval_service = RetrievalEvalService()
    configurations: list[ScreenConfigurationReport] = []
    for mode in manifest.workload.retrieval_modes:
        for limit in manifest.workload.limits:
            key: ScreenKey = (mode, limit)
            artifact_relative_path = Path("runs") / f"{mode}-k{limit}.json"
            artifact_path = destination / artifact_relative_path
            if resume and artifact_path.is_file():
                artifact = ScreenRunArtifact.model_validate_json(
                    artifact_path.read_text(encoding="utf-8")
                )
                _validate_resume_artifact(
                    artifact,
                    manifest=manifest,
                    git_state=git_state,
                    workspace_snapshot_id=validated.workspace.snapshot_id,
                    mode=mode,
                    limit=limit,
                )
                if progress is not None:
                    progress(f"Reused {mode} k={limit}")
            else:
                if progress is not None:
                    progress(f"Starting {mode} k={limit}")
                before = query_cache.stats()
                runtime = build_retrieval_runtime(
                    workspace,
                    mode,
                    limit=limit,
                    config=validated.config,
                    prepared_state_cache=prepared_cache,
                    embedding_service=query_cache,
                )
                evaluation = eval_service.evaluate(
                    cases=selected_cases,
                    search_service=runtime.retrieval_engine,
                    limit=limit,
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
                artifact = ScreenRunArtifact(
                    benchmark=manifest.name,
                    variant_id=manifest.variant.id,
                    git_commit=git_state.commit,
                    workspace_snapshot_id=validated.workspace.snapshot_id,
                    cache=cache_state,
                    cache_reuse_valid=_screen_cache_valid(cache_state),
                    query_cache_hits=after.hits - before.hits,
                    query_cache_misses=after.misses - before.misses,
                    evaluation=evaluation,
                )
                atomic_write_text(
                    artifact_path,
                    artifact.model_dump_json(indent=2) + "\n",
                )
                if progress is not None:
                    progress(
                        f"Completed {mode} k={limit}: "
                        f"hit={evaluation.metrics['hit@k']:.4f}"
                    )

            configurations.append(
                build_screen_configuration(
                    manifest,
                    mode=mode,
                    limit=limit,
                    artifact_path=artifact_relative_path.as_posix(),
                    baseline_reports=parent_reports[key],
                    candidate_report=artifact.evaluation,
                    cache_reuse_valid=artifact.cache_reuse_valid,
                    query_cache_hits=artifact.query_cache_hits,
                    query_cache_misses=artifact.query_cache_misses,
                )
            )

    gates = evaluate_screen_gates(manifest, configurations)
    cache_data = QueryEmbeddingCacheFile.model_validate_json(
        query_cache_path.read_text(encoding="utf-8")
    )
    final_cache_stats = query_cache.stats()
    required_query_keys = {
        _query_key(
            build_query_embedding_text(
                case.query,
                cache_data.query_representation,
            )
        )
        for case in selected_cases
    }
    valid = (
        validated.checks.passed
        and not git_state.dirty
        and all(configuration.cache_reuse_valid for configuration in configurations)
        and set(cache_data.entries) >= required_query_keys
    )
    promotion_applicable = manifest.variant.role == "candidate"
    promoted = (
        valid and all(gate.passed for gate in gates)
        if promotion_applicable
        else None
    )
    report = RetrievalScreenReport(
        benchmark=manifest.name,
        description=manifest.description,
        measured_at=datetime.now(UTC).isoformat(),
        manifest_path=_display_path(source_manifest_path, root),
        manifest_sha256=sha256_file(source_manifest_path, "text_lf"),
        git=git_state,
        runtime=runtime_environment,
        parent_baseline=parent,
        variant=manifest.variant,
        dataset=validated.dataset,
        corpus=validated.corpus,
        workspace=validated.workspace,
        workspace_fingerprints=fingerprints,
        selected_case_ids=manifest.selected_case_ids,
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
        valid=valid,
        promotion_applicable=promotion_applicable,
        promoted=promoted,
    )
    if final_cache_stats != QueryCacheStats(
        hits=report.query_cache.hits,
        misses=report.query_cache.misses,
    ) and not resume:
        raise RuntimeError("Screen query cache statistics are inconsistent")
    atomic_write_text(
        destination / "summary.json",
        report.model_dump_json(indent=2) + "\n",
    )
    return report


def chunk_content_fingerprint(
    chunks: Sequence[Mapping[str, object]],
    corpus_paths: Sequence[str],
) -> str:
    payload = sorted(
        (
            {
                "source_path": _canonical_source_path(
                    _required_string(chunk, "source_path"), corpus_paths
                ),
                "chunk_suffix": _chunk_suffix(
                    _required_string(chunk, "chunk_id")
                ),
                "start_char": _optional_int(chunk.get("start_char")),
                "end_char": _optional_int(chunk.get("end_char")),
                "text_sha256": hash_text(_required_string(chunk, "text")),
            }
            for chunk in chunks
        ),
        key=lambda item: (
            item["source_path"],
            item["chunk_suffix"],
        ),
    )
    return _payload_sha256(payload)


def index_input_fingerprint(
    records: Sequence[VectorIndexRecord],
    corpus_paths: Sequence[str],
) -> str:
    payload = sorted(
        (
            {
                "source_path": _canonical_source_path(
                    record.source_path, corpus_paths
                ),
                "chunk_suffix": _chunk_suffix(record.chunk_id),
                "text_sha256": record.text_hash,
            }
            for record in records
        ),
        key=lambda item: (
            item["source_path"],
            item["chunk_suffix"],
        ),
    )
    return _payload_sha256(payload)


def _validate_screen_inputs(
    manifest: RetrievalScreenManifest,
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
            retrieval_modes=list(manifest.workload.retrieval_modes),
            limits=list(manifest.workload.limits),
            repetitions=3,
            max_quality_metric_spread=0.05,
        ),
    )
    return validate_inputs(
        validation_manifest,
        repository_root,
        workspace,
        manifest.variant.workspace_build_git_commit,
    )


def _load_parent_baseline(
    manifest: RetrievalScreenManifest,
    repository_root: Path,
) -> tuple[
    ScreenResolvedParentBaseline,
    dict[ScreenKey, list[RetrievalEvalReport]],
]:
    summary_path = _resolve_repository_path(
        repository_root,
        manifest.parent_baseline.path,
    )
    digest = sha256_file(summary_path, manifest.parent_baseline.hash_mode)
    if digest != manifest.parent_baseline.sha256:
        raise ValueError("Screen parent baseline summary hash mismatch")
    summary = RetrievalBaselineReport.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    if not summary.passed:
        raise ValueError("Screen parent baseline did not pass")
    if (
        summary.dataset.cases.sha256 != manifest.dataset.sha256
        or summary.dataset.case_count != manifest.dataset.case_count
        or summary.workspace.chunk_count != manifest.ingest.expected_chunk_count
        or summary.workspace.document_count
        != manifest.ingest.expected_document_count
    ):
        raise ValueError("Screen parent baseline inputs do not match manifest")

    reports: dict[ScreenKey, list[RetrievalEvalReport]] = {}
    for mode in manifest.workload.retrieval_modes:
        for limit in manifest.workload.limits:
            configuration = next(
                (
                    item
                    for item in summary.configurations
                    if item.retrieval_mode == mode and item.limit == limit
                ),
                None,
            )
            if configuration is None:
                raise ValueError(
                    f"Parent baseline is missing {mode} k={limit}"
                )
            if len(configuration.trials) != (
                manifest.parent_baseline.required_repetitions
            ):
                raise ValueError("Parent baseline repetition count mismatch")
            trial_reports = []
            for trial in configuration.trials:
                artifact_path = (summary_path.parent / trial.artifact_path).resolve()
                if not artifact_path.is_relative_to(summary_path.parent):
                    raise ValueError(
                        "Parent baseline artifact escapes result directory"
                    )
                artifact = BaselineTrialArtifact.model_validate_json(
                    artifact_path.read_text(encoding="utf-8")
                )
                if (
                    artifact.trial != trial
                    or artifact.workspace_snapshot_id
                    != summary.workspace.snapshot_id
                    or artifact.evaluation.retrieval_mode != mode
                    or artifact.evaluation.limit != limit
                ):
                    raise ValueError("Parent baseline trial artifact mismatch")
                trial_reports.append(artifact.evaluation)
            reports[(mode, limit)] = trial_reports

    return (
        ScreenResolvedParentBaseline(
            summary=BaselineResolvedFile(
                path=_display_path(summary_path, repository_root),
                sha256=digest,
                hash_mode=manifest.parent_baseline.hash_mode,
            ),
            git_commit=summary.git.commit,
            workspace_snapshot_id=summary.workspace.snapshot_id,
        ),
        reports,
    )


def _select_cases(
    validated: ValidatedBaselineInputs,
    selected_case_ids: Sequence[str],
) -> list[RetrievalEvalCase]:
    cases_by_id = {case.id: case for case in validated.cases}
    missing = sorted(set(selected_case_ids) - set(cases_by_id))
    if missing:
        raise ValueError(f"Screen manifest references unknown cases: {missing}")
    return [cases_by_id[case_id] for case_id in selected_case_ids]


def _workspace_fingerprints(
    workspace: LocalWorkspace,
    corpus_paths: Sequence[str],
) -> ScreenWorkspaceFingerprints:
    chunks: list[ChunkRecord] = workspace.read_chunks()
    records = VectorIndexService(workspace).read_index()
    chunk_ids = {str(chunk.get("chunk_id", "")) for chunk in chunks}
    index_ids = {record.chunk_id for record in records}
    if chunk_ids != index_ids:
        raise ValueError("Screen vector index chunk ids do not match workspace chunks")
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


def _screen_cache_valid(cache: BaselineCacheState) -> bool:
    return (
        cache.chunk_loads == 1
        and cache.vector_loads == 1
        and cache.invalidations == 0
    )


def _validate_resume_manifest(
    output_dir: Path,
    manifest: RetrievalScreenManifest,
) -> None:
    copied_path = output_dir / "manifest.json"
    if not copied_path.is_file():
        raise FileNotFoundError(
            f"Screen resume manifest is missing: {copied_path}"
        )
    copied = RetrievalScreenManifest.model_validate_json(
        copied_path.read_text(encoding="utf-8")
    )
    if copied != manifest:
        raise ValueError("Screen resume manifest does not match requested manifest")


def _validate_resume_run_files(
    output_dir: Path,
    manifest: RetrievalScreenManifest,
) -> None:
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return
    expected = {
        f"{mode}-k{limit}.json"
        for mode in manifest.workload.retrieval_modes
        for limit in manifest.workload.limits
    }
    actual = {path.name for path in runs_dir.glob("*.json")}
    unexpected = sorted(actual - expected)
    if unexpected:
        raise ValueError(
            f"Screen resume directory has unexpected artifacts: {unexpected}"
        )


def _validate_resume_artifact(
    artifact: ScreenRunArtifact,
    *,
    manifest: RetrievalScreenManifest,
    git_state: BaselineGitState,
    workspace_snapshot_id: str,
    mode: ScreenMode,
    limit: ScreenLimit,
) -> None:
    evaluation = artifact.evaluation
    if not (
        artifact.benchmark == manifest.name
        and artifact.variant_id == manifest.variant.id
        and artifact.git_commit == git_state.commit
        and artifact.workspace_snapshot_id == workspace_snapshot_id
        and evaluation.retrieval_mode == mode
        and evaluation.limit == limit
        and {result.id for result in evaluation.results}
        == set(manifest.selected_case_ids)
        and artifact.cache_reuse_valid
    ):
        raise ValueError(f"Screen resume artifact mismatch for {mode} k={limit}")


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


def _canonical_source_path(value: str, corpus_paths: Sequence[str]) -> str:
    normalized = value.replace("\\", "/")
    matches = [
        path.replace("\\", "/")
        for path in sorted(corpus_paths, key=len, reverse=True)
        if normalized.casefold().endswith(path.replace("\\", "/").casefold())
    ]
    if not matches:
        raise ValueError(f"Screen source path is outside frozen corpus: {value}")
    return matches[0]


def _chunk_suffix(value: str) -> str:
    suffix = value.rsplit("::", 1)[-1]
    if not suffix or suffix == value:
        raise ValueError(f"Screen chunk id has no canonical suffix: {value}")
    return suffix


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Screen chunk field {key!r} must be a string")
    return value


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to(repository_root):
        raise ValueError(f"Screen path escapes repository root: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"Screen input file not found: {path}")
    return path


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _print_summary(report: RetrievalScreenReport, output_dir: Path) -> None:
    print(f"Retrieval screen: {report.benchmark}")
    print(f"Variant: {report.variant.id} ({report.variant.role})")
    print(f"Cases: {len(report.selected_case_ids)}")
    print(
        "Query cache: "
        f"{report.query_cache.hits} hits, {report.query_cache.misses} misses"
    )
    for configuration in report.configurations:
        print(
            f"{configuration.retrieval_mode}@{configuration.limit}: "
            f"baseline-hit={configuration.baseline_metrics.hit_rate.average:.4f} "
            f"candidate-hit={configuration.candidate_metrics.hit_rate:.4f}"
        )
    print("Gates:")
    for gate in report.gates:
        print(
            f"  {'PASS' if gate.passed else 'FAIL'} {gate.name}: "
            f"{gate.observed} ({gate.requirement})"
        )
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Valid: {report.valid}")
    if report.promotion_applicable:
        print(f"Promoted: {report.promoted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed retrieval representation screening protocol."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to a checked-in retrieval screen manifest.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Prepared full-corpus generation-layout workspace.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for the screen summary and run artifacts.",
    )
    parser.add_argument(
        "--query-cache-source",
        default=None,
        help="Optional compatible frozen query embedding cache to reuse.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a diagnostic dirty-tree run that cannot be promoted.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse completed screen artifacts.",
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
                "Retrieval screening requires a clean Git tree; use "
                "--allow-dirty only for a non-promotable diagnostic run"
            )
        report = run_screen(
            load_manifest(args.manifest),
            manifest_path=args.manifest,
            repository_root=repository_root,
            workspace=LocalWorkspace(args.workspace),
            output_dir=args.output_dir,
            git_state=git_state,
            runtime_environment=collect_runtime_environment(),
            query_cache_source=args.query_cache_source,
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
        print(f"Retrieval screen failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, Path(args.output_dir).resolve())
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
