from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.fragment_packing_development import (
    FROZEN_SELECTOR_INPUT_FIELDS,
    FragmentRunArtifact,
    FragmentRunMetrics,
    _fragment_evidence,
    _load_parent,
    evaluate_fragment_gates,
    load_manifest,
)
from benchmarks.retrieval_baseline import sha256_file
from pydantic import ValidationError

from ragent_forge.core.retrieval.context_fragments import (
    ContextFragment,
    FragmentScoreComponents,
    RankedFragmentCandidate,
)


def _candidate(rank: int, text: str) -> RankedFragmentCandidate:
    return RankedFragmentCandidate(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        source_path="docs/source.pdf",
        source_label="source.pdf",
        page_label="1",
        text=text,
    )


def _fragment(candidate: RankedFragmentCandidate, text: str) -> ContextFragment:
    start = candidate.text.index(text)
    end = start + len(text)
    return ContextFragment(
        rank=candidate.rank,
        chunk_id=candidate.chunk_id,
        source_path=candidate.source_path,
        source_label=candidate.source_label,
        page_label=candidate.page_label,
        start_char=start,
        end_char=end,
        text=text,
        truncated_left=start > 0,
        truncated_right=end < len(candidate.text),
        score=FragmentScoreComponents(
            unique_query_tokens=0,
            query_token_occurrences=0,
            query_bigram_coverage=0,
            signal_token_coverage=0,
        ),
    )


def _metrics(**updates: object) -> FragmentRunMetrics:
    values: dict[str, object] = {
        "parent_hits": 30,
        "scorable_parent_hits": 30,
        "unscorable_parent_hit_case_ids": [],
        "oracle_lost_hit_case_ids": [],
        "selector_lost_hit_case_ids": [],
        "average_selector_context_tokens": 700.0,
        "maximum_selector_context_tokens": 768,
        "average_reachable_evidence_coverage": 0.7,
        "minimum_reachable_evidence_coverage": 0.3,
        "average_oracle_efficiency": 0.8,
        "candidate_representation_violations": [],
        "traceability_violations": [],
        "budget_violations": [],
        "mapping_violations": [],
    }
    values.update(updates)
    return FragmentRunMetrics.model_validate(values)


def _run(mode: str, metrics: FragmentRunMetrics) -> FragmentRunArtifact:
    return FragmentRunArtifact.model_construct(
        retrieval_mode=mode,
        selector_input_fields=list(FROZEN_SELECTOR_INPUT_FIELDS),
        metrics=metrics,
    )


def _failed_gates(
    semantic: FragmentRunArtifact,
    hybrid: FragmentRunArtifact,
) -> set[str]:
    gates, _ = evaluate_fragment_gates(load_manifest(), [semantic, hybrid])
    return {gate.name for gate in gates if not gate.passed}


def test_checked_in_manifest_freezes_parent_hashes_and_allowlist() -> None:
    root = Path(__file__).parents[1]
    manifest = load_manifest()
    specs = [
        manifest.parent,
        manifest.parent.direction_manifest,
        *manifest.parent.runs,
    ]

    assert manifest.variant.selector_input_fields == FROZEN_SELECTOR_INPUT_FIELDS
    assert all(
        sha256_file(root / spec.path, spec.hash_mode) == spec.sha256
        for spec in specs
    )
    parent, runs = _load_parent(manifest, root)
    assert parent.workspace_snapshot_id == (
        "snapshot-20260715T070507Z-e2ed57b0"
    )
    assert set(runs) == {"semantic", "hybrid"}


def test_manifest_rejects_selector_gold_input() -> None:
    payload = load_manifest().model_dump(mode="json")
    variant = payload["variant"]
    assert isinstance(variant, dict)
    fields = variant["selector_input_fields"]
    assert isinstance(fields, list)
    fields[-1] = "gold_evidence"

    with pytest.raises(ValidationError, match="allowlist drifted"):
        type(load_manifest()).model_validate(payload)


def test_fragment_evidence_requires_overlap_inside_relevant_fragment() -> None:
    candidates = [
        _candidate(1, "irrelevant candidate text"),
        _candidate(
            2,
            "prefix quotient space requires a subspace condition suffix",
        ),
    ]
    oracle = [
        _fragment(candidates[0], candidates[0].text),
        _fragment(candidates[1], "quotient space requires a subspace condition"),
    ]
    selector = [
        _fragment(candidates[0], candidates[0].text),
        _fragment(candidates[1], "prefix quotient"),
    ]

    evidence = _fragment_evidence(
        "A quotient space requires a subspace condition.",
        candidates=candidates,
        oracle_fragments=oracle,
        selector_fragments=selector,
        relevant_chunk_ids={"chunk-2"},
        parent_hit=True,
        ngram_size=3,
    )

    assert evidence.scorable
    assert evidence.oracle_retained
    assert not evidence.selector_retained
    assert evidence.selected_evidence_ngrams == 0


def test_fragment_run_metrics_allow_heldout_case_count() -> None:
    metrics = _metrics(case_count=20)

    assert metrics.case_count == 20


def test_each_development_gate_has_an_independent_failure_signal() -> None:
    semantic = _run("semantic", _metrics())
    hybrid = _run("hybrid", _metrics())
    assert _failed_gates(semantic, hybrid) == set()

    assert _failed_gates(
        _run(
            "semantic",
            _metrics(unscorable_parent_hit_case_ids=["case-1"]),
        ),
        hybrid,
    ) == {"oracle_evidence_reachable"}
    assert _failed_gates(
        _run(
            "semantic",
            _metrics(selector_lost_hit_case_ids=["case-1"]),
        ),
        hybrid,
    ) == {"semantic_fragment_hits_retained"}
    assert _failed_gates(
        semantic,
        _run(
            "hybrid",
            _metrics(selector_lost_hit_case_ids=["case-1"]),
        ),
    ) == {"hybrid_fragment_hits_retained"}
    assert _failed_gates(
        _run(
            "semantic",
            _metrics(candidate_representation_violations=["case-1"]),
        ),
        hybrid,
    ) == {"all_candidates_represented"}
    assert _failed_gates(
        _run("semantic", _metrics(traceability_violations=["case-1"])),
        hybrid,
    ) == {"all_fragments_traceable"}
    assert _failed_gates(
        _run("semantic", _metrics(budget_violations=["case-1"])),
        hybrid,
    ) == {"all_contexts_within_budget"}
    assert _failed_gates(
        _run("semantic", _metrics(mapping_violations=["case-1"])),
        hybrid,
    ) == {"complete_evidence_mapping"}
    assert _failed_gates(
        semantic,
        _run("hybrid", _metrics(average_selector_context_tokens=800.0)),
    ) == {"hybrid_context_tokens"}

    isolated = _run("semantic", _metrics())
    isolated.selector_input_fields = [*FROZEN_SELECTOR_INPUT_FIELDS[:-1], "gold"]
    assert _failed_gates(isolated, hybrid) == {"selector_gold_isolation"}
