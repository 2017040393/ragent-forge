from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.retrieval_baseline import (
    DEFAULT_MANIFEST_PATH,
    load_manifest,
    run_baseline,
    sha256_file,
)
from pydantic import ValidationError

from ragent_forge.app.services.evaluation.baseline import (
    BaselineCacheState,
    BaselineCorpusSpec,
    BaselineCutoffMetrics,
    BaselineDatasetSpec,
    BaselineFileSpec,
    BaselineGitState,
    BaselineIngestSpec,
    BaselineLatencySummary,
    BaselinePackageVersions,
    BaselineRuntimeEnvironment,
    BaselineTrialReport,
    BaselineWorkloadSpec,
    RetrievalBaselineManifest,
    build_configuration_report,
)
from ragent_forge.app.services.ingest_service import IngestService
from ragent_forge.infrastructure.local_workspace import LocalWorkspace


def test_text_hash_is_independent_of_checkout_line_endings(tmp_path: Path) -> None:
    lf_path = tmp_path / "lf.txt"
    crlf_path = tmp_path / "crlf.txt"
    lf_path.write_bytes(b"first\nsecond\n")
    crlf_path.write_bytes(b"first\r\nsecond\r\n")

    assert sha256_file(lf_path, "text_lf") == sha256_file(crlf_path, "text_lf")
    assert sha256_file(lf_path, "binary") != sha256_file(crlf_path, "binary")


def test_checked_in_baseline_manifest_and_input_hashes_are_valid() -> None:
    repository_root = Path(__file__).parents[1]
    manifest = load_manifest()

    assert manifest.workload.repetitions == 3
    assert manifest.workload.limits == [5, 10, 20]
    assert manifest.workload.exact_ranking_modes == ["lexical", "bm25"]
    assert manifest.workload.max_quality_metric_spread == 0.05
    assert manifest.embedding is not None
    assert manifest.embedding.timeout_seconds == 60
    assert sha256_file(
        repository_root / manifest.dataset.path,
        manifest.dataset.hash_mode,
    ) == manifest.dataset.sha256
    assert sha256_file(
        repository_root / manifest.dataset.manifest.path,
        manifest.dataset.manifest.hash_mode,
    ) == manifest.dataset.manifest.sha256
    for source in manifest.corpus.files:
        assert sha256_file(
            repository_root / source.path,
            source.hash_mode,
        ) == source.sha256
    assert (
        repository_root / "benchmarks/retrieval_baseline_manifest.json"
        == DEFAULT_MANIFEST_PATH
    )


def test_manifest_requires_three_repetitions_and_valid_limits() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 3"):
        BaselineWorkloadSpec(
            retrieval_modes=["lexical"],
            limits=[5],
            repetitions=2,
            max_quality_metric_spread=0.05,
        )

    with pytest.raises(ValidationError, match="limits must be unique"):
        BaselineWorkloadSpec(
            retrieval_modes=["lexical"],
            limits=[5, 5],
            repetitions=3,
            max_quality_metric_spread=0.05,
        )

    with pytest.raises(ValidationError, match="must be included"):
        BaselineWorkloadSpec(
            retrieval_modes=["lexical"],
            limits=[5],
            repetitions=3,
            exact_ranking_modes=["bm25"],
            max_quality_metric_spread=0.05,
        )


def test_sparse_baseline_writes_isolated_trial_artifacts(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    knowledge_root = repository_root / "knowledge"
    knowledge_root.mkdir(parents=True)
    source_path = knowledge_root / "rag.md"
    source_path.write_text(
        "Agent memory retrieval keeps project facts inspectable.",
        encoding="utf-8",
    )
    cases_path = repository_root / "cases.jsonl"
    cases_path.write_text(
        "".join(
            json.dumps(case) + "\n"
            for case in [
                {
                    "id": "case-001",
                    "query": "agent retrieval",
                    "expected_source_paths": ["knowledge/rag.md"],
                },
                {
                    "id": "case-002",
                    "query": "project facts",
                    "expected_source_paths": ["knowledge/rag.md"],
                },
            ]
        ),
        encoding="utf-8",
    )
    dataset_manifest_path = repository_root / "cases.manifest.json"
    dataset_manifest_path.write_text("{}\n", encoding="utf-8")

    workspace = LocalWorkspace(tmp_path / ".ragent")
    ingest_result = IngestService(chunk_size=1000, chunk_overlap=0).ingest(
        knowledge_root
    )
    workspace.commit_ingest_generation(ingest_result)

    manifest = RetrievalBaselineManifest(
        name="test-post-architecture-baseline",
        description="Small deterministic sparse baseline.",
        dataset=BaselineDatasetSpec(
            path="cases.jsonl",
            sha256=sha256_file(cases_path, "text_lf"),
            hash_mode="text_lf",
            case_count=2,
            manifest=BaselineFileSpec(
                path="cases.manifest.json",
                sha256=sha256_file(dataset_manifest_path, "text_lf"),
                hash_mode="text_lf",
            ),
        ),
        corpus=BaselineCorpusSpec(
            root="knowledge",
            files=[
                BaselineFileSpec(
                    path="knowledge/rag.md",
                    sha256=sha256_file(source_path, "text_lf"),
                    hash_mode="text_lf",
                )
            ],
        ),
        ingest=BaselineIngestSpec(
            chunk_size=1000,
            chunk_overlap=0,
            expected_document_count=1,
            expected_chunk_count=1,
        ),
        workload=BaselineWorkloadSpec(
            retrieval_modes=["lexical"],
            limits=[1, 2],
            repetitions=3,
            exact_ranking_modes=["lexical"],
            max_quality_metric_spread=0.05,
        ),
    )
    manifest_path = repository_root / "baseline-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    output_dir = repository_root / "results"

    report = run_baseline(
        manifest,
        manifest_path=manifest_path,
        repository_root=repository_root,
        workspace=workspace,
        output_dir=output_dir,
        git_state=BaselineGitState(
            commit="a" * 40,
            branch="test",
            dirty=False,
        ),
        workspace_build_git_commit="b" * 40,
        runtime_environment=BaselineRuntimeEnvironment(
            python="3.11.0",
            implementation="CPython",
            platform="test-platform",
            machine="test-machine",
            processor="test-processor",
            packages=BaselinePackageVersions(
                ragent_forge="0.2.0",
                pydantic="2.0.0",
                httpx="0.27.0",
                pdfplumber="0.11.0",
            ),
        ),
    )

    configuration = next(
        item for item in report.configurations if item.limit == 1
    )
    assert report.passed is True
    assert report.trial_git_commits == ["a" * 40]
    assert report.workspace.layout == "generation"
    assert report.workspace.build_git_commit == "b" * 40
    assert report.workspace.snapshot_id == workspace.current_snapshot_id()
    assert configuration.ranking_stable is True
    assert configuration.quality_stable is True
    assert configuration.warm_latency_ms.sample_count == 3
    assert configuration.metrics_by_cutoff[1].hit_rate.average == 1.0
    assert (
        configuration.metrics_by_cutoff[1].avg_selected_context_chars.average > 0
    )
    assert all(trial.cache_reuse_valid for trial in configuration.trials)
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert len(list((output_dir / "runs").glob("*.json"))) == 6

    missing_artifact = output_dir / "runs" / "lexical-k2-trial-03.json"
    missing_artifact.unlink()
    (output_dir / "summary.json").unlink()
    progress: list[str] = []
    resumed = run_baseline(
        manifest,
        manifest_path=manifest_path,
        repository_root=repository_root,
        workspace=workspace,
        output_dir=output_dir,
        git_state=BaselineGitState(
            commit="c" * 40,
            branch="test",
            dirty=False,
        ),
        workspace_build_git_commit="b" * 40,
        runtime_environment=report.runtime,
        resume=True,
        progress=progress.append,
    )

    assert resumed.passed is True
    assert resumed.trial_git_commits == ["a" * 40, "c" * 40]
    assert len([message for message in progress if message.startswith("Reused")]) == 5
    completed_messages = [
        message for message in progress if message.startswith("Completed")
    ]
    assert len(completed_messages) == 1
    assert len(list((output_dir / "runs").glob("*.json"))) == 6


def test_dense_configuration_accepts_bounded_quality_variation() -> None:
    trials = [
        _trial_with_hit_rate(1, "a" * 64, 0.58),
        _trial_with_hit_rate(2, "b" * 64, 0.60),
        _trial_with_hit_rate(3, "c" * 64, 0.62),
    ]

    report = build_configuration_report(
        trials,
        require_identical_rankings=False,
        max_quality_metric_spread=0.05,
    )

    assert report.ranking_stable is False
    assert report.quality_stable is True
    assert report.passed is True
    assert report.metrics_by_cutoff[5].hit_rate.model_dump() == {
        "average": 0.6,
        "minimum": 0.58,
        "maximum": 0.62,
        "spread": 0.04,
    }

    exact_report = build_configuration_report(
        trials,
        require_identical_rankings=True,
        max_quality_metric_spread=0.05,
    )
    strict_report = build_configuration_report(
        trials,
        require_identical_rankings=False,
        max_quality_metric_spread=0.03,
    )

    assert exact_report.passed is False
    assert strict_report.quality_stable is False
    assert strict_report.passed is False


def _trial_with_hit_rate(
    repetition: int,
    fingerprint: str,
    hit_rate: float,
) -> BaselineTrialReport:
    latency = BaselineLatencySummary(
        sample_count=2,
        average_ms=2.5,
        p50_ms=2.5,
        p95_ms=2.95,
    )
    return BaselineTrialReport(
        repetition=repetition,
        retrieval_mode="semantic",
        retrieval_method="semantic_cosine_similarity",
        limit=5,
        artifact_path=f"runs/semantic-k5-trial-{repetition:02d}.json",
        result_fingerprint_sha256=fingerprint,
        metrics_by_cutoff={
            5: BaselineCutoffMetrics(
                cutoff=5,
                hit_rate=hit_rate,
                recall=0.3,
                precision=0.12,
                ndcg=0.4,
                mrr=0.35,
                passed_count=round(hit_rate * 50),
                failed_count=50 - round(hit_rate * 50),
                avg_selected_context_chars=3000,
                avg_selected_context_tokens=750,
            )
        },
        cold_start_latency_ms=10,
        warm_latency_samples_ms=[2, 3],
        warm_latency_ms=latency,
        cold_stage_latency_ms={},
        warm_stage_latency_ms={},
        cache=BaselineCacheState(
            snapshot_id="snapshot-1",
            chunk_loads=1,
            vector_loads=1,
            warm_hits=2,
            invalidations=0,
        ),
        cache_reuse_valid=True,
    )
