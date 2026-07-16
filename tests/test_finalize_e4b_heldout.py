from __future__ import annotations

from pathlib import Path

from benchmarks.finalize_e4b_heldout import (
    CorrectedReviewDecision,
    FinalizedHeldoutDatasetManifest,
    finalize_heldout_dataset,
    load_review_manifest,
)
from benchmarks.retrieval_baseline import sha256_file

from ragent_forge.app.services.evaluation.cases import load_cases


def test_manual_review_covers_all_cases_and_freezes_one_correction() -> None:
    review = load_review_manifest()

    assert len(review.decisions) == 20
    corrections = [
        decision
        for decision in review.decisions
        if isinstance(decision, CorrectedReviewDecision)
    ]
    assert [decision.case_id for decision in corrections] == [
        "e4b-heldout-000017"
    ]


def test_finalizer_validates_archive_and_applies_reviewed_correction(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    dataset_path = tmp_path / "cases.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = finalize_heldout_dataset(
        load_review_manifest(),
        review_path=root / "benchmarks/e4b_heldout_manual_review.json",
        repository_root=root,
        output_dataset=dataset_path,
        output_manifest=manifest_path,
    )

    cases = load_cases(dataset_path)
    corrected = next(case for case in cases if case.id == "e4b-heldout-000017")
    assert manifest.valid is True
    assert manifest.manual_pass_count == 19
    assert manifest.manual_correction_count == 1
    assert "10^158" in corrected.metadata["reference_answer"]
    assert "c^n" not in corrected.metadata["reference_answer"]


def test_checked_in_finalized_dataset_matches_manifest() -> None:
    root = Path(__file__).parents[1]
    path = root / "examples/eval/e4b_heldout_confirmation.manifest.json"
    manifest = FinalizedHeldoutDatasetManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )

    assert manifest.valid is True
    assert sha256_file(root / manifest.dataset_path, "text_lf") == (
        manifest.dataset_sha256
    )
    assert len(load_cases(root / manifest.dataset_path)) == 20
