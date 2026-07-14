from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ragent_forge.app.models import AppConfig
from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.evaluation.baseline import (
    BaselineCacheState,
    BaselineFileSpec,
    BaselineGitState,
    BaselineIndexState,
    BaselineInputChecks,
    BaselinePackageVersions,
    BaselineResolvedDataset,
    BaselineResolvedFile,
    BaselineRuntimeEnvironment,
    BaselineTrialArtifact,
    BaselineWorkspaceState,
    HashMode,
    RetrievalBaselineManifest,
    RetrievalBaselineReport,
    build_configuration_report,
    build_trial_report,
)
from ragent_forge.app.services.evaluation.contracts import RetrievalEvalCase
from ragent_forge.app.services.evaluation.runner import RetrievalEvalService
from ragent_forge.app.services.prepared_retrieval import PreparedStateCache
from ragent_forge.app.services.search_service import tokenize
from ragent_forge.app.services.vector_index_service import VectorIndexService
from ragent_forge.composition import build_retrieval_runtime
from ragent_forge.core.retrieval.types import RetrievalMode
from ragent_forge.infrastructure.local_workspace import LocalWorkspace
from ragent_forge.infrastructure.storage import atomic_write_text

DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "retrieval_baseline_manifest.json"
)


@dataclass(frozen=True)
class ValidatedBaselineInputs:
    cases: list[RetrievalEvalCase]
    config: AppConfig
    dataset: BaselineResolvedDataset
    corpus: list[BaselineResolvedFile]
    workspace: BaselineWorkspaceState
    checks: BaselineInputChecks


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> RetrievalBaselineManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RetrievalBaselineManifest.model_validate(payload)


def run_baseline(
    manifest: RetrievalBaselineManifest,
    *,
    manifest_path: str | Path,
    repository_root: str | Path,
    workspace: LocalWorkspace,
    output_dir: str | Path,
    git_state: BaselineGitState,
    runtime_environment: BaselineRuntimeEnvironment,
    workspace_build_git_commit: str | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> RetrievalBaselineReport:
    root = Path(repository_root).resolve()
    source_manifest_path = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and not resume:
        raise FileExistsError(
            f"Baseline output directory already exists: {destination}"
        )
    if resume and not destination.is_dir():
        raise FileNotFoundError(
            f"Baseline resume directory does not exist: {destination}"
        )

    resolved_workspace_build_commit = workspace_build_git_commit or git_state.commit
    validated = validate_inputs(
        manifest,
        root,
        workspace,
        resolved_workspace_build_commit,
    )
    if resume:
        _validate_resume_manifest(destination, manifest)
        _validate_resume_run_files(destination, manifest)
    else:
        destination.mkdir(parents=True)
        atomic_write_text(
            destination / "manifest.json",
            manifest.model_dump_json(indent=2) + "\n",
        )

    measured_at = datetime.now(UTC).isoformat()
    eval_service = RetrievalEvalService()
    configurations = []
    trial_git_commits: set[str] = set()
    for mode in manifest.workload.retrieval_modes:
        for limit in manifest.workload.limits:
            trials = []
            for repetition in range(1, manifest.workload.repetitions + 1):
                artifact_relative_path = Path("runs") / (
                    f"{mode}-k{limit}-trial-{repetition:02d}.json"
                )
                artifact_path = destination / artifact_relative_path
                if resume and artifact_path.is_file():
                    artifact = _load_resume_artifact(artifact_path)
                    _validate_resume_artifact(
                        artifact,
                        manifest=manifest,
                        workspace_snapshot_id=validated.workspace.snapshot_id,
                        workspace_build_git_commit=resolved_workspace_build_commit,
                        mode=mode,
                        limit=limit,
                        repetition=repetition,
                        artifact_relative_path=artifact_relative_path,
                    )
                    trials.append(artifact.trial)
                    trial_git_commits.add(artifact.git_commit)
                    if progress is not None:
                        progress(
                            f"Reused {mode} k={limit} trial {repetition} "
                            f"from {artifact.git_commit[:7]}"
                        )
                    continue
                if progress is not None:
                    progress(
                        f"Starting {mode} k={limit} trial {repetition}/"
                        f"{manifest.workload.repetitions}"
                    )
                cache = PreparedStateCache(tokenize)
                runtime = build_retrieval_runtime(
                    workspace,
                    mode,
                    limit=limit,
                    config=validated.config,
                    prepared_state_cache=cache,
                )
                evaluation = eval_service.evaluate(
                    cases=validated.cases,
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
                cache_stats = cache.stats()
                trial = build_trial_report(
                    evaluation,
                    repetition=repetition,
                    cutoffs=[limit],
                    artifact_path=artifact_relative_path.as_posix(),
                    cache=BaselineCacheState(
                        snapshot_id=cache_stats.snapshot_id,
                        chunk_loads=cache_stats.chunk_loads,
                        vector_loads=cache_stats.vector_loads,
                        warm_hits=cache_stats.warm_hits,
                        invalidations=cache_stats.invalidations,
                    ),
                )
                artifact = BaselineTrialArtifact(
                    benchmark=manifest.name,
                    git_commit=git_state.commit,
                    workspace_build_git_commit=resolved_workspace_build_commit,
                    workspace_snapshot_id=validated.workspace.snapshot_id,
                    trial=trial,
                    evaluation=evaluation,
                )
                atomic_write_text(
                    artifact_path,
                    artifact.model_dump_json(indent=2) + "\n",
                )
                trials.append(trial)
                trial_git_commits.add(git_state.commit)
                if progress is not None:
                    progress(
                        f"Completed {mode} k={limit} trial {repetition}: "
                        f"cold={trial.cold_start_latency_ms:.4f} ms, "
                        f"warm-p95={trial.warm_latency_ms.p95_ms:.4f} ms"
                    )
            configurations.append(
                build_configuration_report(
                    trials,
                    require_identical_rankings=(
                        mode in manifest.workload.exact_ranking_modes
                    ),
                    max_quality_metric_spread=(
                        manifest.workload.max_quality_metric_spread
                    ),
                )
            )

    report = RetrievalBaselineReport(
        benchmark=manifest.name,
        description=manifest.description,
        measured_at=measured_at,
        manifest_path=_display_path(source_manifest_path, root),
        manifest_sha256=sha256_file(source_manifest_path, "text_lf"),
        git=git_state,
        trial_git_commits=sorted(trial_git_commits),
        runtime=runtime_environment,
        dataset=validated.dataset,
        corpus=validated.corpus,
        workspace=validated.workspace,
        input_checks=validated.checks,
        configurations=configurations,
        passed=(
            validated.checks.passed
            and not git_state.dirty
            and all(configuration.passed for configuration in configurations)
        ),
    )
    atomic_write_text(
        destination / "summary.json",
        report.model_dump_json(indent=2) + "\n",
    )
    return report


def _validate_resume_manifest(
    output_dir: Path,
    manifest: RetrievalBaselineManifest,
) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Baseline resume manifest is missing: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = RetrievalBaselineManifest.model_validate(payload)
    if existing != manifest:
        raise ValueError("Baseline resume manifest does not match requested manifest")


def _validate_resume_run_files(
    output_dir: Path,
    manifest: RetrievalBaselineManifest,
) -> None:
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return
    expected_names = {
        f"{mode}-k{limit}-trial-{repetition:02d}.json"
        for mode in manifest.workload.retrieval_modes
        for limit in manifest.workload.limits
        for repetition in range(1, manifest.workload.repetitions + 1)
    }
    actual_names = {path.name for path in runs_dir.glob("*.json")}
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        raise ValueError(
            "Baseline resume directory contains unexpected trial artifacts: "
            f"{unexpected}"
        )


def _load_resume_artifact(path: Path) -> BaselineTrialArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BaselineTrialArtifact.model_validate(payload)


def _validate_resume_artifact(
    artifact: BaselineTrialArtifact,
    *,
    manifest: RetrievalBaselineManifest,
    workspace_snapshot_id: str,
    workspace_build_git_commit: str,
    mode: RetrievalMode,
    limit: int,
    repetition: int,
    artifact_relative_path: Path,
) -> None:
    trial = artifact.trial
    evaluation = artifact.evaluation
    matches = (
        artifact.benchmark == manifest.name
        and artifact.workspace_snapshot_id == workspace_snapshot_id
        and artifact.workspace_build_git_commit == workspace_build_git_commit
        and trial.retrieval_mode == mode
        and trial.limit == limit
        and trial.repetition == repetition
        and trial.artifact_path == artifact_relative_path.as_posix()
        and trial.cache.snapshot_id == workspace_snapshot_id
        and trial.cache_reuse_valid
        and set(trial.metrics_by_cutoff) == {limit}
        and evaluation.retrieval_mode == mode
        and evaluation.limit == limit
        and evaluation.case_count == manifest.dataset.case_count
        and _normalized_path(evaluation.cases_path)
        == _normalized_path(manifest.dataset.path)
    )
    if not matches:
        raise ValueError(
            "Baseline resume artifact does not match requested trial: "
            f"{artifact_relative_path.as_posix()}"
        )


def _normalized_path(value: str) -> str:
    return value.strip().replace("\\", "/")


def sha256_file(path: str | Path, hash_mode: HashMode) -> str:
    file_path = Path(path)
    if hash_mode == "binary":
        content = file_path.read_bytes()
    else:
        text = file_path.read_text(encoding="utf-8")
        content = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def collect_git_state(
    repository_root: str | Path,
    *,
    ignored_untracked_roots: Sequence[str | Path] = (),
) -> BaselineGitState:
    root = Path(repository_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "branch", "--show-current") or None
    unstaged = _git_has_diff(root, cached=False)
    staged = _git_has_diff(root, cached=True)
    ignored_roots = [Path(path).resolve() for path in ignored_untracked_roots]
    untracked_paths = [
        (root / line).resolve()
        for line in _git_output(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
        ).splitlines()
        if line
    ]
    untracked = any(
        not any(path.is_relative_to(ignored) for ignored in ignored_roots)
        for path in untracked_paths
    )
    return BaselineGitState(
        commit=commit,
        branch=branch,
        dirty=unstaged or staged or untracked,
    )


def collect_runtime_environment() -> BaselineRuntimeEnvironment:
    return BaselineRuntimeEnvironment(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        packages=BaselinePackageVersions(
            ragent_forge=_package_version("ragent-forge"),
            pydantic=_package_version("pydantic"),
            httpx=_package_version("httpx"),
            pdfplumber=_package_version("pdfplumber"),
        ),
    )


def validate_inputs(
    manifest: RetrievalBaselineManifest,
    repository_root: Path,
    workspace: LocalWorkspace,
    workspace_build_git_commit: str,
) -> ValidatedBaselineInputs:
    eval_service = RetrievalEvalService()
    dataset_file = _validate_file(manifest.dataset, repository_root)
    dataset_manifest_file = _validate_file(
        manifest.dataset.manifest,
        repository_root,
    )
    cases_path = _resolve_repository_path(repository_root, manifest.dataset.path)
    cases = eval_service.load_cases(cases_path)
    if len(cases) != manifest.dataset.case_count:
        raise ValueError(
            "Baseline dataset case count mismatch: expected "
            f"{manifest.dataset.case_count}, found {len(cases)}"
        )

    source_root = _resolve_repository_path(repository_root, manifest.corpus.root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Baseline corpus root not found: {source_root}")
    corpus = []
    for file_spec in manifest.corpus.files:
        source_file = _resolve_repository_path(repository_root, file_spec.path)
        if not source_file.is_relative_to(source_root):
            raise ValueError(
                f"Baseline corpus file is outside corpus root: {file_spec.path}"
            )
        corpus.append(_validate_file(file_spec, repository_root))
    if not workspace.uses_generation_layout():
        raise ValueError("Baseline workspace must use generation storage layout")

    snapshot_id = workspace.current_snapshot_id()
    snapshot_manifest = workspace.read_snapshot_manifest()
    if snapshot_id is None or snapshot_manifest is None:
        raise ValueError("Baseline workspace snapshot manifest is missing")
    if snapshot_manifest.snapshot_id != snapshot_id:
        raise ValueError("Baseline workspace snapshot manifest does not match current")
    if Path(snapshot_manifest.source_path).resolve() != source_root:
        raise ValueError("Baseline snapshot source does not match corpus root")

    ingest_summary = workspace.read_ingest_summary()
    ingest_metadata = _required_mapping(ingest_summary, "metadata")
    chunk_size = _required_int(ingest_metadata, "chunk_size")
    chunk_overlap = _required_int(ingest_metadata, "chunk_overlap")
    document_count = _required_int(ingest_summary, "document_count")
    chunk_count = _required_int(ingest_summary, "chunk_count")
    source_path = _required_string(ingest_summary, "source_path")
    if Path(source_path).resolve() != source_root:
        raise ValueError(
            "Baseline workspace source mismatch: expected "
            f"{source_root}, found {source_path}"
        )
    if (
        chunk_size != manifest.ingest.chunk_size
        or chunk_overlap != manifest.ingest.chunk_overlap
    ):
        raise ValueError("Baseline workspace chunking configuration mismatch")
    if (
        document_count != manifest.ingest.expected_document_count
        or chunk_count != manifest.ingest.expected_chunk_count
        or snapshot_manifest.chunk_count != chunk_count
    ):
        raise ValueError("Baseline workspace document or chunk count mismatch")

    config = ConfigService(workspace).load()
    dense_requested = bool(
        {"semantic", "hybrid"}.intersection(manifest.workload.retrieval_modes)
    )
    index_state: BaselineIndexState | None = None
    if dense_requested:
        if manifest.embedding is None:
            raise ValueError("Dense baseline modes require embedding expectations")
        if not workspace.has_vector_index():
            raise ValueError("Dense baseline modes require a vector index")
        if (
            config.embedding.provider != manifest.embedding.provider
            or config.embedding.model != manifest.embedding.model
            or config.embedding.batch_size != manifest.embedding.batch_size
            or config.embedding.timeout_seconds != manifest.embedding.timeout_seconds
        ):
            raise ValueError("Baseline embedding configuration mismatch")
        index_manifest = VectorIndexService(workspace).read_manifest()
        index_state = BaselineIndexState(
            embedding_provider=_required_string(
                index_manifest,
                "embedding_provider",
            ),
            embedding_model=_required_string(index_manifest, "embedding_model"),
            embedding_dim=_required_int(index_manifest, "embedding_dim"),
            chunk_count=_required_int(index_manifest, "chunk_count"),
            snapshot_id=_required_string(index_manifest, "snapshot_id"),
        )
        if (
            index_state.embedding_provider != manifest.embedding.provider
            or index_state.embedding_model != manifest.embedding.model
            or index_state.embedding_dim != manifest.embedding.dimensions
            or index_state.chunk_count != chunk_count
            or index_state.snapshot_id != snapshot_id
        ):
            raise ValueError("Baseline vector index manifest mismatch")
        required_artifacts = {
            "chunks",
            "ingest_summary",
            "vector_index",
            "vector_index_manifest",
        }
        if set(snapshot_manifest.artifacts) != required_artifacts:
            raise ValueError("Baseline snapshot does not contain the complete index")

    checks = BaselineInputChecks(
        dataset_hash=True,
        dataset_manifest_hash=True,
        source_hashes=True,
        case_count=True,
        generation_layout=True,
        snapshot_manifest=True,
        ingest_configuration=True,
        workspace_counts=True,
        embedding_configuration=True,
        vector_index=True,
    )
    workspace_state = BaselineWorkspaceState(
        root=str(workspace.root_path.resolve()),
        schema_version=snapshot_manifest.schema_version,
        build_git_commit=workspace_build_git_commit,
        snapshot_id=snapshot_id,
        source_path=str(source_root),
        document_count=document_count,
        chunk_count=chunk_count,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_provider=(config.embedding.provider if dense_requested else None),
        embedding_model=(config.embedding.model if dense_requested else None),
        embedding_batch_size=(
            config.embedding.batch_size if dense_requested else None
        ),
        embedding_timeout_seconds=(
            config.embedding.timeout_seconds if dense_requested else None
        ),
        index=index_state,
    )
    return ValidatedBaselineInputs(
        cases=cases,
        config=config,
        dataset=BaselineResolvedDataset(
            cases=dataset_file,
            manifest=dataset_manifest_file,
            case_count=len(cases),
        ),
        corpus=corpus,
        workspace=workspace_state,
        checks=checks,
    )


def _validate_file(
    file_spec: BaselineFileSpec,
    repository_root: Path,
) -> BaselineResolvedFile:
    path = _resolve_repository_path(repository_root, file_spec.path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline input file not found: {path}")
    digest = sha256_file(path, file_spec.hash_mode)
    if digest != file_spec.sha256:
        raise ValueError(
            f"Baseline input hash mismatch for {file_spec.path}: "
            f"expected {file_spec.sha256}, found {digest}"
        )
    return BaselineResolvedFile(
        path=_display_path(path, repository_root),
        sha256=digest,
        hash_mode=file_spec.hash_mode,
    )


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to(repository_root):
        raise ValueError(f"Baseline path escapes repository root: {value}")
    return path


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _required_mapping(
    payload: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Baseline metadata field {key!r} must be an object")
    return {str(item_key): item_value for item_key, item_value in value.items()}


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Baseline metadata field {key!r} must be an integer")
    return value


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Baseline metadata field {key!r} must be a string")
    return value


def _git_output(repository_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_has_diff(repository_root: Path, *, cached: bool) -> bool:
    args = ["git", "diff", "--quiet"]
    if cached:
        args.insert(2, "--cached")
    completed = subprocess.run(args, cwd=repository_root, check=False)
    if completed.returncode not in {0, 1}:
        raise RuntimeError("Unable to inspect Git working tree state")
    return completed.returncode == 1


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _print_summary(report: RetrievalBaselineReport, output_dir: Path) -> None:
    print(f"Retrieval baseline: {report.benchmark}")
    print(f"Git commit: {report.git.commit}")
    print(f"Trial commits: {', '.join(report.trial_git_commits)}")
    print(f"Workspace snapshot: {report.workspace.snapshot_id}")
    print(f"Cases: {report.dataset.case_count}")
    print()
    print("mode      k   rank-stable  quality-stable  cold-p95-ms  warm-p95-ms")
    for configuration in report.configurations:
        print(
            f"{configuration.retrieval_mode:<9} "
            f"{configuration.limit:<3} "
            f"{str(configuration.ranking_stable):<12} "
            f"{str(configuration.quality_stable):<15} "
            f"{configuration.cold_start_latency_ms.p95_ms:<12.4f} "
            f"{configuration.warm_latency_ms.p95_ms:.4f}"
        )
        for cutoff, metrics in configuration.metrics_by_cutoff.items():
            print(
                f"  {configuration.retrieval_mode}@{cutoff:<2} "
                f"hit={metrics.hit_rate.average:.4f} "
                f"recall={metrics.recall.average:.4f} "
                f"precision={metrics.precision.average:.4f} "
                f"nDCG={metrics.ndcg.average:.4f} "
                f"MRR={metrics.mrr.average:.4f}"
            )
    print()
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Passed: {report.passed}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a reproducible cold/warm retrieval quality baseline."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the checked-in retrieval baseline manifest.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Prepared generation-layout workspace to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for summary and per-trial JSON artifacts.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a diagnostic run from a Git tree with content changes.",
    )
    parser.add_argument(
        "--workspace-build-commit",
        default=None,
        help=(
            "Full Git commit used to ingest and index the workspace; defaults "
            "to the evaluation commit."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse completed trials in an existing output directory.",
    )
    args = parser.parse_args(argv)

    try:
        repository_root = Path(
            _git_output(Path.cwd(), "rev-parse", "--show-toplevel")
        )
        git_state = collect_git_state(
            repository_root,
            ignored_untracked_roots=([args.output_dir] if args.resume else ()),
        )
        if git_state.dirty and not args.allow_dirty:
            raise ValueError(
                "Formal baseline requires a clean Git tree; commit or remove "
                "content changes, or use --allow-dirty for a diagnostic run"
            )
        report = run_baseline(
            load_manifest(args.manifest),
            manifest_path=args.manifest,
            repository_root=repository_root,
            workspace=LocalWorkspace(args.workspace),
            output_dir=args.output_dir,
            git_state=git_state,
            runtime_environment=collect_runtime_environment(),
            workspace_build_git_commit=args.workspace_build_commit,
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
        print(f"Retrieval baseline failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, Path(args.output_dir).resolve())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
