from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.fragment_packing_confirmation import (
    FragmentPackingConfirmationReport,
    HeldoutDatasetValidation,
    _query_cache_report,
    _query_key,
    _validate_development,
    _validate_heldout_dataset,
    evaluate_confirmation_gates,
    load_manifest,
)
from benchmarks.fragment_packing_development import (
    FROZEN_SELECTOR_INPUT_FIELDS,
    FragmentCaseResult,
    FragmentEvidenceMetrics,
    FragmentRunArtifact,
    FragmentRunMetrics,
)
from benchmarks.retrieval_baseline import sha256_file
from benchmarks.retrieval_screen import QueryEmbeddingCacheFile
from pydantic import ValidationError

from ragent_forge.app.services.evaluation.cases import load_cases
from ragent_forge.app.services.evaluation.screening import ScreenRunArtifact
from ragent_forge.core.retrieval.representations import build_query_embedding_text


def _evidence(
    *,
    parent_hit: bool,
    coverage: float | None = None,
    efficiency: float | None = None,
) -> FragmentEvidenceMetrics:
    return FragmentEvidenceMetrics(
        parent_hit=parent_hit,
        scorable=parent_hit,
        reachable_evidence_ngrams=10 if parent_hit else 0,
        selected_evidence_ngrams=7 if parent_hit else 0,
        oracle_evidence_ngrams=8 if parent_hit else 0,
        reachable_evidence_coverage=coverage if parent_hit else None,
        oracle_efficiency=efficiency if parent_hit else None,
        oracle_retained=True,
        selector_retained=True,
    )


def _cases(
    *,
    parent_hits: int = 8,
    low_coverage_case: bool = False,
) -> list[FragmentCaseResult]:
    cases: list[FragmentCaseResult] = []
    for index in range(20):
        hit = index < parent_hits
        coverage = 0.2 if low_coverage_case and index == 0 else 0.7
        cases.append(
            FragmentCaseResult.model_construct(
                case_id=f"case-{index + 1}",
                parent_relevant_ranks=([1] if hit else []),
                evidence=_evidence(
                    parent_hit=hit,
                    coverage=coverage,
                    efficiency=0.85,
                ),
            )
        )
    return cases


def _metrics(**updates: object) -> FragmentRunMetrics:
    values: dict[str, object] = {
        "case_count": 20,
        "parent_hits": 8,
        "scorable_parent_hits": 8,
        "unscorable_parent_hit_case_ids": [],
        "oracle_lost_hit_case_ids": [],
        "selector_lost_hit_case_ids": [],
        "average_selector_context_tokens": 700.0,
        "maximum_selector_context_tokens": 768,
        "average_reachable_evidence_coverage": 0.7,
        "minimum_reachable_evidence_coverage": 0.7,
        "average_oracle_efficiency": 0.85,
        "candidate_representation_violations": [],
        "traceability_violations": [],
        "budget_violations": [],
        "mapping_violations": [],
    }
    values.update(updates)
    return FragmentRunMetrics.model_validate(values)


def _run(
    mode: str,
    *,
    metrics: FragmentRunMetrics | None = None,
    cases: list[FragmentCaseResult] | None = None,
) -> FragmentRunArtifact:
    return FragmentRunArtifact.model_construct(
        retrieval_mode=mode,
        selector_input_fields=list(FROZEN_SELECTOR_INPUT_FIELDS),
        metrics=metrics or _metrics(),
        cases=cases or _cases(),
    )


def _heldout(*, passed: bool = True) -> HeldoutDatasetValidation:
    return HeldoutDatasetValidation.model_construct(
        canonical_query_duplicates=[],
        canonical_span_overlaps=[],
        provenance_violations=([] if passed else ["drift"]),
        passed=passed,
    )


def _failed_gates(
    semantic: FragmentRunArtifact,
    hybrid: FragmentRunArtifact,
    *,
    heldout: HeldoutDatasetValidation | None = None,
) -> set[str]:
    gates, _ = evaluate_confirmation_gates(
        load_manifest(),
        [semantic, hybrid],
        dataset_holdout=heldout or _heldout(),
    )
    return {gate.name for gate in gates if not gate.passed}


def test_checked_in_manifest_and_dataset_provenance_are_frozen() -> None:
    root = Path(__file__).parents[1]
    manifest = load_manifest()
    specs = [
        manifest.development.manifest,
        manifest.development.summary,
        manifest.dataset,
        manifest.dataset.manifest,
        manifest.canonical_dataset,
        *manifest.corpus.files,
    ]

    assert all(
        sha256_file(root / spec.path, spec.hash_mode) == spec.sha256
        for spec in specs
    )
    _, development = _validate_development(manifest, root)
    validation = _validate_heldout_dataset(
        manifest,
        repository_root=root,
        cases=load_cases(root / manifest.dataset.path),
    )
    assert development.implementation_git_commit == (
        "ccc930ae27740fca1da706c08487cc66cbb5493f"
    )
    assert validation.passed is True
    assert validation.unique_span_count == 10


def test_manifest_rejects_post_registration_threshold_change() -> None:
    payload = load_manifest().model_dump(mode="json")
    confirmation = payload["confirmation"]
    assert isinstance(confirmation, dict)
    confirmation["minimum_average_evidence_coverage"] = 0.59

    with pytest.raises(ValidationError, match="thresholds drifted"):
        type(load_manifest()).model_validate(payload)


def test_all_thirteen_confirmation_gates_have_independent_failure_signals() -> None:
    semantic = _run("semantic")
    hybrid = _run("hybrid")
    gates, _ = evaluate_confirmation_gates(
        load_manifest(),
        [semantic, hybrid],
        dataset_holdout=_heldout(),
    )
    assert len(gates) == 13
    assert _failed_gates(semantic, hybrid) == set()

    assert _failed_gates(semantic, hybrid, heldout=_heldout(passed=False)) == {
        "dataset_is_held_out"
    }
    assert _failed_gates(
        _run("semantic", metrics=_metrics(parent_hits=7)), hybrid
    ) == {"minimum_parent_hits"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(unscorable_parent_hit_case_ids=["case-1"]),
        ),
        hybrid,
    ) == {"oracle_evidence_reachable"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(selector_lost_hit_case_ids=["case-1"]),
        ),
        hybrid,
    ) == {"fragment_hits_retained"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(average_reachable_evidence_coverage=0.59),
        ),
        hybrid,
    ) == {"average_evidence_coverage"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(minimum_reachable_evidence_coverage=0.2),
            cases=_cases(low_coverage_case=True),
        ),
        hybrid,
    ) == {"minimum_evidence_coverage"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(average_oracle_efficiency=0.79),
        ),
        hybrid,
    ) == {"oracle_efficiency"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(candidate_representation_violations=["case-1"]),
        ),
        hybrid,
    ) == {"all_candidates_represented"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(traceability_violations=["case-1"]),
        ),
        hybrid,
    ) == {"all_fragments_traceable"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(budget_violations=["case-1"]),
        ),
        hybrid,
    ) == {"all_contexts_within_budget"}
    assert _failed_gates(
        _run(
            "semantic",
            metrics=_metrics(mapping_violations=["case-1"]),
        ),
        hybrid,
    ) == {"complete_evidence_mapping"}
    assert _failed_gates(
        semantic,
        _run(
            "hybrid",
            metrics=_metrics(average_selector_context_tokens=800.0),
        ),
    ) == {"hybrid_context_tokens"}

    isolated = _run("semantic")
    isolated.selector_input_fields = [
        *FROZEN_SELECTOR_INPUT_FIELDS[:-1],
        "gold_evidence",
    ]
    assert _failed_gates(isolated, hybrid) == {"selector_gold_isolation"}


def test_query_cache_contract_requires_independence_and_hybrid_reuse(
    tmp_path: Path,
) -> None:
    manifest = load_manifest()
    cases = load_cases(Path(__file__).parents[1] / manifest.dataset.path)
    keys = {
        _query_key(
            build_query_embedding_text(
                case.query,
                manifest.ranking.query_embedding_representation,
            )
        )
        for case in cases
    }
    cache = QueryEmbeddingCacheFile(
        provider=manifest.embedding.provider,
        model=manifest.embedding.model,
        query_representation=manifest.ranking.query_embedding_representation,
        embedding_dim=manifest.embedding.dimensions,
        entries={key: [] for key in keys},
    )
    cache_path = tmp_path / "query_embeddings.json"
    cache_path.write_text(cache.model_dump_json(indent=2) + "\n", encoding="utf-8")
    semantic = ScreenRunArtifact.model_construct(
        query_cache_hits=0,
        query_cache_misses=20,
    )
    hybrid = ScreenRunArtifact.model_construct(
        query_cache_hits=20,
        query_cache_misses=0,
    )

    report = _query_cache_report(
        manifest,
        cache_path=cache_path,
        cache=cache,
        cases=cases,
        ranking_artifacts={"semantic": semantic, "hybrid": hybrid},
    )

    assert report.independent is True
    assert report.hybrid_reuse_complete is True
    assert report.valid is True


def test_checked_in_heldout_result_is_valid_but_not_confirmed() -> None:
    root = Path(__file__).parents[1]
    path = (
        root
        / "benchmarks/results/fragment-packing/E4b-heldout-41cb038/summary.json"
    )
    report = FragmentPackingConfirmationReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )

    assert sha256_file(path, "text_lf") == (
        "236a889f201f65a2e4acfc072f44998a27141de86eabfc6081d30756ec421b0d"
    )
    assert report.git.commit == "41cb038e5d5c313e8c58d90930361ca7da74b92d"
    assert report.valid is True
    assert report.confirmed is False
    assert {gate.name for gate in report.gates if not gate.passed} == {
        "minimum_parent_hits",
        "average_evidence_coverage",
        "minimum_evidence_coverage",
    }
    assert report.query_cache.independent is True
    assert report.query_cache.hybrid_reuse_complete is True
