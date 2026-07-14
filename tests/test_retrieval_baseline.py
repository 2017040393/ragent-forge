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
    BaselineCorpusSpec,
    BaselineDatasetSpec,
    BaselineFileSpec,
    BaselineGitState,
    BaselineIngestSpec,
    BaselinePackageVersions,
    BaselineRuntimeEnvironment,
    BaselineWorkloadSpec,
    RetrievalBaselineManifest,
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
    assert manifest.workload.max_limit == 20
    assert manifest.workload.cutoffs == [5, 10, 20]
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


def test_manifest_requires_three_repetitions_and_bounded_cutoffs() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 3"):
        BaselineWorkloadSpec(
            retrieval_modes=["lexical"],
            max_limit=5,
            cutoffs=[5],
            repetitions=2,
        )

    with pytest.raises(ValidationError, match="must not exceed max_limit"):
        BaselineWorkloadSpec(
            retrieval_modes=["lexical"],
            max_limit=5,
            cutoffs=[5, 10],
            repetitions=3,
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
            max_limit=2,
            cutoffs=[1, 2],
            repetitions=3,
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

    configuration = report.configurations[0]
    assert report.passed is True
    assert report.workspace.layout == "generation"
    assert report.workspace.snapshot_id == workspace.current_snapshot_id()
    assert configuration.quality_stable is True
    assert configuration.warm_latency_ms.sample_count == 3
    assert configuration.metrics_by_cutoff[1].hit_rate == 1.0
    assert configuration.metrics_by_cutoff[1].avg_selected_context_chars > 0
    assert all(trial.cache_reuse_valid for trial in configuration.trials)
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert len(list((output_dir / "runs").glob("*.json"))) == 3
