from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.retrieval_baseline import sha256_file
from benchmarks.retrieval_screen import (
    DEFAULT_MANIFEST_PATH,
    CachedQueryEmbeddingService,
    QueryEmbeddingCacheFile,
    chunk_content_fingerprint,
    index_input_fingerprint,
    load_manifest,
)
from pydantic import ValidationError

from ragent_forge.app.models import EmbeddingResult
from ragent_forge.app.services.evaluation.contracts import (
    FailureType,
    RetrievalEvalCaseResult,
    RetrievalEvalReport,
)
from ragent_forge.app.services.evaluation.screening import (
    RetrievalScreenManifest,
    ScreenLimit,
    ScreenMode,
    ScreenWorkloadSpec,
    build_screen_configuration,
    evaluate_screen_gates,
)
from ragent_forge.app.services.vector_index_service import (
    VectorIndexRecord,
    hash_text,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(texts)
        return EmbeddingResult(
            provider_name="openai_embeddings",
            model="test-model",
            embeddings=[
                [float(len(text)), float(index)]
                for index, text in enumerate(texts)
            ],
        )


def test_checked_in_screen_manifest_freezes_parent_and_case_groups() -> None:
    repository_root = Path(__file__).parents[1]
    manifest = load_manifest()

    assert manifest.variant.id == "E0-raw-text"
    assert manifest.variant.role == "baseline"
    assert manifest.workload.repetitions == 1
    assert manifest.workload.retrieval_modes == ["semantic", "hybrid"]
    assert manifest.workload.limits == [5, 20]
    assert len(manifest.selected_case_ids) == 16
    assert len(manifest.gated_case_ids) == 14
    assert {group.role for group in manifest.case_groups} == {
        "stable_control",
        "semantic_opportunity",
        "wrong_section_challenge",
        "hard_miss",
        "boundary_canary",
    }
    assert sha256_file(
        repository_root / manifest.parent_baseline.path,
        manifest.parent_baseline.hash_mode,
    ) == manifest.parent_baseline.sha256
    assert sha256_file(
        repository_root / manifest.dataset.path,
        manifest.dataset.hash_mode,
    ) == manifest.dataset.sha256
    assert (
        repository_root / "benchmarks/retrieval_screen_manifest.json"
    ) == DEFAULT_MANIFEST_PATH


def test_screen_contract_is_separate_from_formal_three_trial_workload() -> None:
    with pytest.raises(ValidationError, match="semantic and hybrid"):
        ScreenWorkloadSpec(
            retrieval_modes=["semantic", "semantic"],
            limits=[5, 20],
            repetitions=1,
        )

    payload = load_manifest().model_dump(mode="json")
    case_groups = payload["case_groups"]
    assert isinstance(case_groups, list)
    final_group = case_groups[-1]
    first_group = case_groups[0]
    assert isinstance(final_group, dict)
    assert isinstance(first_group, dict)
    final_ids = final_group["case_ids"]
    first_ids = first_group["case_ids"]
    assert isinstance(final_ids, list)
    assert isinstance(first_ids, list)
    final_ids[0] = first_ids[0]

    with pytest.raises(ValidationError, match="globally unique"):
        RetrievalScreenManifest.model_validate(payload)


def test_query_embedding_cache_reuses_frozen_vectors(tmp_path: Path) -> None:
    first_client = FakeEmbeddingClient()
    first_path = tmp_path / "first-cache.json"
    first = CachedQueryEmbeddingService(
        first_client,
        cache_path=first_path,
        provider="openai_embeddings",
        model="test-model",
        query_representation="raw_query_v1",
        embedding_dim=2,
    )

    initial = first.embed_texts(["alpha", "beta"])
    repeated = first.embed_texts(["alpha"])

    assert len(initial.embeddings) == 2
    assert repeated.embeddings == [initial.embeddings[0]]
    assert first.stats().hits == 1
    assert first.stats().misses == 2
    assert first_client.calls == [["alpha", "beta"]]
    cache = QueryEmbeddingCacheFile.model_validate_json(
        first_path.read_text(encoding="utf-8")
    )
    assert len(cache.entries) == 2

    second_client = FakeEmbeddingClient()
    second = CachedQueryEmbeddingService(
        second_client,
        cache_path=tmp_path / "second-cache.json",
        source_path=first_path,
        provider="openai_embeddings",
        model="test-model",
        query_representation="raw_query_v1",
        embedding_dim=2,
    )

    assert second.embed_texts(["beta"]).embeddings == [initial.embeddings[1]]
    assert second.stats().hits == 1
    assert second.stats().misses == 0
    assert second_client.calls == []


def test_workspace_fingerprints_are_path_portable() -> None:
    corpus_paths = ["examples/knowledge/rag.md"]
    first_chunks: list[dict[str, object]] = [
        {
            "chunk_id": "C:\\repo\\examples\\knowledge\\rag.md::chunk-0000",
            "source_path": "C:\\repo\\examples\\knowledge\\rag.md",
            "text": "retrieval text",
            "start_char": 0,
            "end_char": 14,
        }
    ]
    second_chunks: list[dict[str, object]] = [
        {
            **first_chunks[0],
            "chunk_id": "/repo/examples/knowledge/rag.md::chunk-0000",
            "source_path": "/repo/examples/knowledge/rag.md",
        }
    ]
    record = VectorIndexRecord(
        chunk_id="/repo/examples/knowledge/rag.md::chunk-0000",
        document_id="/repo/examples/knowledge/rag.md",
        source_path="/repo/examples/knowledge/rag.md",
        embedding_provider="openai_embeddings",
        embedding_model="test-model",
        embedding_dim=2,
        embedding=[0.1, 0.2],
        text_hash=hash_text("retrieval text"),
    )

    assert chunk_content_fingerprint(first_chunks, corpus_paths) == (
        chunk_content_fingerprint(second_chunks, corpus_paths)
    )
    assert len(index_input_fingerprint([record], corpus_paths)) == 64


def test_promotion_gates_use_case_transitions_not_slice_average() -> None:
    manifest = load_manifest()
    stable_ids = {
        case_id
        for group in manifest.case_groups
        if group.role in {"stable_control", "semantic_opportunity"}
        for case_id in group.case_ids
    }
    challenge_ids = {
        case_id
        for group in manifest.case_groups
        if group.role in {"wrong_section_challenge", "hard_miss"}
        for case_id in group.case_ids
    }
    wrong_ids = next(
        set(group.case_ids)
        for group in manifest.case_groups
        if group.role == "wrong_section_challenge"
    )
    opportunity_ids = next(
        set(group.case_ids)
        for group in manifest.case_groups
        if group.role == "semantic_opportunity"
    )
    opportunity_hybrid_hit = {"v0-2-baseline-000041"}
    baseline_hits: dict[tuple[ScreenMode, ScreenLimit], set[str]] = {
        ("semantic", 5): stable_ids,
        ("semantic", 20): stable_ids
        | {"v0-2-baseline-000003", "v0-2-baseline-000017"},
        ("hybrid", 5): (
            (stable_ids - opportunity_ids)
            | opportunity_hybrid_hit
            | wrong_ids
        ),
        ("hybrid", 20): stable_ids | wrong_ids,
    }
    candidate_hits = {key: set(value) for key, value in baseline_hits.items()}
    candidate_hits[("semantic", 5)].add("v0-2-baseline-000003")
    candidate_hits[("semantic", 20)].update(
        {"v0-2-baseline-000031", "v0-2-baseline-000005"}
    )
    assert candidate_hits[("semantic", 5)] & challenge_ids

    configurations = []
    for mode in manifest.workload.retrieval_modes:
        for limit in manifest.workload.limits:
            key: tuple[ScreenMode, ScreenLimit] = (mode, limit)
            baseline_reports = [
                _evaluation_report(manifest, mode, limit, baseline_hits[key])
                for _trial in range(3)
            ]
            candidate_report = _evaluation_report(
                manifest,
                mode,
                limit,
                candidate_hits[key],
            )
            configurations.append(
                build_screen_configuration(
                    manifest,
                    mode=mode,
                    limit=limit,
                    artifact_path=f"runs/{mode}-k{limit}.json",
                    baseline_reports=baseline_reports,
                    candidate_report=candidate_report,
                    cache_reuse_valid=True,
                    query_cache_hits=16,
                    query_cache_misses=0,
                )
            )

    gates = evaluate_screen_gates(manifest, configurations)

    assert all(gate.passed for gate in gates)
    challenge_gate = next(
        gate for gate in gates if gate.name == "semantic_challenge_gain"
    )
    assert challenge_gate.observed == "new_top5=1,new_top20=2"


def _evaluation_report(
    manifest: RetrievalScreenManifest,
    mode: ScreenMode,
    limit: ScreenLimit,
    hit_ids: set[str],
) -> RetrievalEvalReport:
    results = [
        _case_result(case_id, passed=case_id in hit_ids, limit=limit)
        for case_id in manifest.selected_case_ids
    ]
    return RetrievalEvalReport(
        retrieval_mode=mode,
        retrieval_method=(
            "semantic_cosine_similarity" if mode == "semantic" else "hybrid_rrf"
        ),
        limit=limit,
        case_count=len(results),
        passed_count=sum(result.passed for result in results),
        failed_count=sum(not result.passed for result in results),
        metrics={},
        cases_path=manifest.dataset.path,
        workspace=".ragent/test",
        results=results,
    )


def _case_result(
    case_id: str,
    *,
    passed: bool,
    limit: int,
) -> RetrievalEvalCaseResult:
    rank = 1 if passed else None
    failure_type: FailureType | None = None if passed else "wrong_section"
    relevant_ranks = [1] if passed else []
    context_chars = limit * 100
    return RetrievalEvalCaseResult(
        id=case_id,
        query=f"query for {case_id}",
        passed=passed,
        rank=rank,
        reciprocal_rank=1.0 if passed else 0.0,
        matched_by="chunk_id" if passed else "none",
        failure_type=failure_type,
        failure_reason=None if passed else "expected section was not retrieved",
        expected_chunk_ids=[f"expected::{case_id}"],
        expected_source_paths=[],
        actual_chunk_ids=[f"actual::{case_id}::{index}" for index in range(limit)],
        actual_source_paths=["examples/knowledge/rag.md"],
        top_results=[
            {"rank": index, "text_chars": 100}
            for index in range(1, limit + 1)
        ],
        retrieved_count=limit,
        expected_chunk_count=1,
        relevant_retrieved_count=1 if passed else 0,
        relevant_result_ranks=relevant_ranks,
        recall=1.0 if passed else 0.0,
        precision=(1 / limit) if passed else 0.0,
        ndcg=1.0 if passed else 0.0,
        evidence_coverage=1.0 if passed else 0.0,
        mapping_coverage=1.0,
        context_evidence_density=(1 / limit) if passed else 0.0,
        duplicate_context_ratio=0.0,
        retrieval_latency_ms=1.0,
        retrieved_context_chars=context_chars,
        estimated_context_tokens=context_chars // 4,
    )
