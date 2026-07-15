from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import pytest
from benchmarks.direction_confirmation import (
    DEFAULT_MANIFEST_PATH,
    DirectionCaseComparison,
    DirectionConfigurationReport,
    DirectionParentCaseOutcome,
    DirectionSelectionMetrics,
    _build_ranking_checkpoint,
    _load_parent_baseline,
    _validate_query_cache_lineage,
    _validate_resume_ranking,
    build_selection_artifact,
    evaluate_direction_gates,
    load_manifest,
)
from benchmarks.retrieval_baseline import sha256_file
from benchmarks.retrieval_screen import QueryEmbeddingCacheFile
from pydantic import ValidationError

from ragent_forge.app.services.evaluation.baseline import (
    BaselineCacheState,
    BaselineCutoffMetricDistribution,
    BaselineCutoffMetrics,
    BaselineMetricDistribution,
)
from ragent_forge.app.services.evaluation.contracts import (
    RetrievalEvalCaseResult,
    RetrievalEvalReport,
)
from ragent_forge.app.services.evaluation.screening import ScreenRunArtifact

Mode = Literal["semantic", "hybrid"]


def _metric_distribution(average: float) -> BaselineMetricDistribution:
    return BaselineMetricDistribution(
        average=average,
        minimum=average,
        maximum=average,
        spread=0.0,
    )


def _parent_metrics(mode: Mode) -> BaselineCutoffMetricDistribution:
    hit_rate = 0.26 if mode == "semantic" else 0.64
    context_tokens = 315.0 if mode == "semantic" else 724.5
    return BaselineCutoffMetricDistribution(
        cutoff=5,
        hit_rate=_metric_distribution(hit_rate),
        recall=_metric_distribution(hit_rate),
        precision=_metric_distribution(hit_rate / 5),
        ndcg=_metric_distribution(hit_rate),
        mrr=_metric_distribution(hit_rate),
        passed_count=_metric_distribution(hit_rate * 50),
        failed_count=_metric_distribution((1 - hit_rate) * 50),
        avg_selected_context_chars=_metric_distribution(context_tokens * 4),
        avg_selected_context_tokens=_metric_distribution(context_tokens),
    )


def _case_result(
    case_id: str,
    *,
    relevant_rank: int | None = 1,
    text_lengths: list[int] | None = None,
    mapping_coverage: float = 1.0,
) -> RetrievalEvalCaseResult:
    lengths = text_lengths or [100, 100, 100, 100, 100]
    chunk_ids = [f"{case_id}-chunk-{rank}" for rank in range(1, 6)]
    relevant_ranks = [] if relevant_rank is None else [relevant_rank]
    hit = relevant_rank is not None
    return RetrievalEvalCaseResult(
        id=case_id,
        query=f"query {case_id}",
        passed=hit,
        rank=relevant_rank,
        reciprocal_rank=(0.0 if relevant_rank is None else 1 / relevant_rank),
        matched_by=("chunk_id" if hit else "none"),
        failure_type=(None if hit else "wrong_section"),
        failure_reason=(None if hit else "not found"),
        expected_chunk_ids=[
            "expected" if relevant_rank is None else chunk_ids[relevant_rank - 1]
        ],
        expected_source_paths=[],
        actual_chunk_ids=chunk_ids,
        actual_source_paths=["source.md"] * 5,
        top_results=[
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "text_chars": text_length,
            }
            for rank, (chunk_id, text_length) in enumerate(
                zip(chunk_ids, lengths, strict=True),
                start=1,
            )
        ],
        retrieved_count=5,
        expected_chunk_count=1,
        relevant_retrieved_count=len(relevant_ranks),
        relevant_result_ranks=relevant_ranks,
        recall=float(hit),
        precision=(0.2 if hit else 0.0),
        ndcg=(1.0 if relevant_rank == 1 else 0.0),
        mapping_coverage=mapping_coverage,
        context_evidence_density=0.0,
        duplicate_context_ratio=0.0,
        retrieval_latency_ms=1.0,
        retrieved_context_chars=sum(lengths),
        estimated_context_tokens=math.ceil(sum(lengths) / 4),
    )


def _evaluation(mode: Mode) -> RetrievalEvalReport:
    results = [_case_result(f"case-{index:02d}") for index in range(50)]
    return RetrievalEvalReport(
        retrieval_mode=mode,
        retrieval_method=(
            "semantic_cosine_similarity" if mode == "semantic" else "hybrid_rrf"
        ),
        limit=5,
        case_count=50,
        passed_count=50,
        failed_count=0,
        metrics={},
        cases_path="cases.jsonl",
        workspace="workspace",
        results=results,
    )


def _ranking_artifact(mode: Mode) -> ScreenRunArtifact:
    manifest = load_manifest()
    return ScreenRunArtifact(
        benchmark=manifest.name,
        variant_id=manifest.candidate.id,
        git_commit="a" * 40,
        workspace_snapshot_id="snapshot-test",
        cache=BaselineCacheState(
            snapshot_id="snapshot-test",
            chunk_loads=1,
            vector_loads=1,
            warm_hits=49,
            invalidations=0,
        ),
        cache_reuse_valid=True,
        query_cache_hits=50,
        query_cache_misses=0,
        evaluation=_evaluation(mode),
    )


def _selection_metrics(
    *,
    average_tokens: float = 700.0,
) -> DirectionSelectionMetrics:
    return DirectionSelectionMetrics(
        case_count=50,
        parent_ranking_hits=25,
        selected_hits=25,
        lost_hit_case_ids=[],
        average_selected_chunks=3.0,
        average_selected_context_chars=average_tokens * 4,
        average_estimated_context_tokens=average_tokens,
        maximum_estimated_context_tokens=768,
        below_minimum_case_ids=[],
        over_budget_case_ids=[],
        invalid_prefix_case_ids=[],
        incomplete_mapping_case_ids=[],
    )


def _configuration(mode: Mode) -> DirectionConfigurationReport:
    parent = _parent_metrics(mode)
    delta = 0.04 if mode == "semantic" else 0.0
    hit_rate = parent.hit_rate.average + delta
    cases = [
        DirectionCaseComparison(
            case_id=f"case-{index:02d}",
            parent=DirectionParentCaseOutcome(
                hit_count=2,
                ranks=[1, 1, None],
                failure_types=[None, None, "wrong_section"],
            ),
            candidate_hit=True,
            candidate_rank=1,
            candidate_failure_type=None,
            transition="retained",
            context_hit_retained=True,
        )
        for index in range(50)
    ]
    return DirectionConfigurationReport(
        retrieval_mode=mode,
        ranking_artifact_path=f"ranking-runs/{mode}-k5.json",
        ranking_checkpoint_path=f"ranking-runs/{mode}-k5.checkpoint.json",
        selection_artifact_path=f"selection-runs/{mode}-k5.json",
        result_fingerprint_sha256="b" * 64,
        parent_metrics=parent,
        candidate_ranking_metrics=BaselineCutoffMetrics(
            cutoff=5,
            hit_rate=hit_rate,
            recall=hit_rate,
            precision=hit_rate / 5,
            ndcg=hit_rate,
            mrr=hit_rate,
            passed_count=round(hit_rate * 50),
            failed_count=50 - round(hit_rate * 50),
            avg_selected_context_chars=2800.0,
            avg_selected_context_tokens=700.0,
        ),
        hit_rate_delta=delta,
        selection_metrics=_selection_metrics(),
        query_cache_hits=50,
        query_cache_misses=0,
        cache_reuse_valid=True,
        cases=cases,
    )


def _failed_gate_names(
    semantic: DirectionConfigurationReport,
    hybrid: DirectionConfigurationReport,
) -> set[str]:
    return {
        gate.name
        for gate in evaluate_direction_gates(load_manifest(), [semantic, hybrid])
        if not gate.passed
    }


def test_checked_in_manifest_freezes_all_direction_inputs() -> None:
    root = Path(__file__).parents[1]
    manifest = load_manifest()
    file_specs = [
        manifest.parent_baseline,
        *manifest.parent_baseline.runs,
        manifest.dataset,
        manifest.dataset.manifest,
        *manifest.corpus.files,
        manifest.query_cache_seed,
    ]

    assert (
        root / "benchmarks/direction_confirmation_manifest_e4a.json"
        == DEFAULT_MANIFEST_PATH
    )
    assert manifest.candidate.max_context_tokens == 768
    assert manifest.candidate.characters_per_token == 4
    assert manifest.promotion.min_semantic_hit_rate_delta == 0.04
    assert all(
        sha256_file(root / spec.path, spec.hash_mode) == spec.sha256
        for spec in file_specs
    )

    parent, reports, metrics = _load_parent_baseline(manifest, root)
    assert len(parent.runs) == 6
    assert len(reports["semantic"]) == len(reports["hybrid"]) == 3
    assert metrics["semantic"].hit_rate.average == 0.26
    assert metrics["hybrid"].hit_rate.average == 0.64


def test_manifest_rejects_duplicate_parent_runs_and_policy_drift() -> None:
    payload = load_manifest().model_dump(mode="json")
    parent = payload["parent_baseline"]
    assert isinstance(parent, dict)
    runs = parent["runs"]
    assert isinstance(runs, list)
    runs[1] = runs[0]

    with pytest.raises(ValidationError, match="six unique Top-5 runs"):
        type(load_manifest()).model_validate(payload)

    payload = load_manifest().model_dump(mode="json")
    candidate = payload["candidate"]
    assert isinstance(candidate, dict)
    candidate["max_context_tokens"] = 769
    with pytest.raises(ValidationError):
        type(load_manifest()).model_validate(payload)


def test_selection_is_a_budgeted_prefix_and_exposes_hit_loss() -> None:
    manifest = load_manifest()
    ranking = _ranking_artifact("semantic")
    ranking.evaluation.results[0] = _case_result(
        "case-00",
        relevant_rank=3,
        text_lengths=[1600, 1600, 100, 100, 100],
    )

    selection = build_selection_artifact(manifest, ranking_artifact=ranking)

    assert selection.metrics.lost_hit_case_ids == ["case-00"]
    assert selection.cases[0].selected_ranks == [1]
    assert selection.cases[0].selected_chunk_ids == (
        selection.cases[0].ranked_chunk_ids[:1]
    )
    assert selection.metrics.maximum_estimated_context_tokens <= 768


def test_each_frozen_gate_has_an_independent_failure_signal() -> None:
    semantic = _configuration("semantic")
    hybrid = _configuration("hybrid")
    assert _failed_gate_names(semantic, hybrid) == set()

    assert _failed_gate_names(
        semantic.model_copy(update={"hit_rate_delta": 0.02}), hybrid
    ) == {"semantic_hit_direction"}
    assert _failed_gate_names(
        semantic, hybrid.model_copy(update={"hit_rate_delta": -0.02})
    ) == {"hybrid_hit_nonnegative"}

    missed_case = semantic.cases[0].model_copy(
        update={"candidate_failure_type": "missed_source"}
    )
    assert _failed_gate_names(
        semantic.model_copy(update={"cases": [missed_case, *semantic.cases[1:]]}),
        hybrid,
    ) == {"no_new_missed_source"}

    semantic_loss = semantic.selection_metrics.model_copy(
        update={"lost_hit_case_ids": ["case-00"]}
    )
    assert _failed_gate_names(
        semantic.model_copy(update={"selection_metrics": semantic_loss}), hybrid
    ) == {"semantic_context_hits_retained"}

    hybrid_loss = hybrid.selection_metrics.model_copy(
        update={"lost_hit_case_ids": ["case-00"]}
    )
    assert _failed_gate_names(
        semantic, hybrid.model_copy(update={"selection_metrics": hybrid_loss})
    ) == {"hybrid_context_hits_retained"}

    over_ratio = hybrid.selection_metrics.model_copy(
        update={"average_estimated_context_tokens": 800.0}
    )
    assert _failed_gate_names(
        semantic, hybrid.model_copy(update={"selection_metrics": over_ratio})
    ) == {"hybrid_context_tokens"}

    invalid_selection = semantic.selection_metrics.model_copy(
        update={"below_minimum_case_ids": ["case-00"]}
    )
    assert _failed_gate_names(
        semantic.model_copy(update={"selection_metrics": invalid_selection}), hybrid
    ) == {"context_selection_invariants"}


def test_query_cache_lineage_preserves_seed_vectors() -> None:
    root = Path(__file__).parents[1]
    manifest = load_manifest()
    seed_path = root / manifest.query_cache_seed.path
    seed = QueryEmbeddingCacheFile.model_validate_json(
        seed_path.read_text(encoding="utf-8")
    )
    cache = seed.model_copy(
        update={
            "source_path": str(seed_path.resolve()),
            "source_sha256": manifest.query_cache_seed.sha256,
        }
    )
    _validate_query_cache_lineage(cache, manifest, root)

    key = next(iter(cache.entries))
    changed_entries = dict(cache.entries)
    changed_entries[key] = [0.0] * manifest.embedding.dimensions
    changed = cache.model_copy(update={"entries": changed_entries})
    with pytest.raises(ValueError, match="seed vectors changed"):
        _validate_query_cache_lineage(changed, manifest, root)


def test_resume_checkpoint_detects_changed_ranking_or_cache(tmp_path: Path) -> None:
    manifest = load_manifest()
    ranking = _ranking_artifact("semantic")
    ranking_path = tmp_path / "semantic-k5.json"
    query_cache_path = tmp_path / "query_embeddings.json"
    ranking_path.write_text(
        ranking.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    query_cache_path.write_text(
        QueryEmbeddingCacheFile(
            provider="test",
            model="test",
            query_representation="instructed_query_v1",
            embedding_dim=1,
            entries={"key": [1.0]},
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    checkpoint = _build_ranking_checkpoint(
        ranking,
        manifest=manifest,
        mode="semantic",
        ranking_path=ranking_path,
        query_cache_path=query_cache_path,
    )

    _validate_resume_ranking(
        ranking,
        checkpoint=checkpoint,
        manifest=manifest,
        mode="semantic",
        git_commit=ranking.git_commit,
        workspace_snapshot_id=ranking.workspace_snapshot_id,
        ranking_path=ranking_path,
        query_cache_path=query_cache_path,
    )

    query_cache_path.write_text(
        query_cache_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resume ranking mismatch"):
        _validate_resume_ranking(
            ranking,
            checkpoint=checkpoint,
            manifest=manifest,
            mode="semantic",
            git_commit=ranking.git_commit,
            workspace_snapshot_id=ranking.workspace_snapshot_id,
            ranking_path=ranking_path,
            query_cache_path=query_cache_path,
        )
