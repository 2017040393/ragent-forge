from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import BaseModel, Field, model_validator

from benchmarks.generate_e4b_heldout import HeldoutGenerationReport
from benchmarks.retrieval_baseline import sha256_file
from ragent_forge.app.services.evaluation.baseline import BaselineFileSpec
from ragent_forge.app.services.evaluation.cases import load_cases
from ragent_forge.infrastructure.storage import atomic_write_text

DEFAULT_REVIEW_PATH = Path(__file__).with_name("e4b_heldout_manual_review.json")
DEFAULT_DATASET_PATH = Path("examples/eval/e4b_heldout_confirmation.jsonl")
DEFAULT_MANIFEST_PATH = Path(
    "examples/eval/e4b_heldout_confirmation.manifest.json"
)


class ReviewDecisionBase(BaseModel):
    case_id: str = Field(pattern=r"^e4b-heldout-[0-9]{6}$")
    original_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_reference_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PassedReviewDecision(ReviewDecisionBase):
    status: Literal["passed"] = "passed"


class CorrectedReviewDecision(ReviewDecisionBase):
    status: Literal["corrected"] = "corrected"
    replacement_query: str = Field(min_length=1)
    replacement_reference_answer: str = Field(min_length=1)
    reason_code: Literal["ungrounded_generated_detail"]
    note: str = Field(min_length=1)


ReviewDecision: TypeAlias = Annotated[
    PassedReviewDecision | CorrectedReviewDecision,
    Field(discriminator="status"),
]


class HeldoutManualReviewManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: Literal["pre-v0.3-e4b-heldout-manual-review"]
    reviewed_on: Literal["2026-07-16"]
    generation_summary: BaselineFileSpec
    raw_dataset: BaselineFileSpec
    canonical_dataset: BaselineFileSpec
    criteria: list[str] = Field(min_length=4, max_length=4)
    decisions: list[ReviewDecision] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def _complete_unique_review(self) -> HeldoutManualReviewManifest:
        case_ids = [decision.case_id for decision in self.decisions]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("manual review case IDs must be unique")
        return self


class FinalizedHeldoutDatasetManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: Literal["pre-v0.3-e4b-heldout-confirmation-dataset"]
    generation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    generation_summary: BaselineFileSpec
    raw_dataset: BaselineFileSpec
    review_manifest: BaselineFileSpec
    canonical_dataset: BaselineFileSpec
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: Literal[20] = 20
    unique_query_count: Literal[20] = 20
    canonical_query_duplicates: list[str]
    manual_pass_count: int = Field(ge=0)
    manual_correction_count: int = Field(ge=0)
    corrected_case_ids: list[str]
    valid: bool


def load_review_manifest(
    path: str | Path = DEFAULT_REVIEW_PATH,
) -> HeldoutManualReviewManifest:
    return HeldoutManualReviewManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def finalize_heldout_dataset(
    review: HeldoutManualReviewManifest,
    *,
    review_path: str | Path,
    repository_root: str | Path,
    output_dataset: str | Path,
    output_manifest: str | Path,
) -> FinalizedHeldoutDatasetManifest:
    root = Path(repository_root).resolve()
    review_source = Path(review_path).resolve()
    dataset_path = Path(output_dataset).resolve()
    manifest_path = Path(output_manifest).resolve()
    if dataset_path.exists() or manifest_path.exists():
        raise FileExistsError("held-out finalized output already exists")
    _require_within_repository(review_source, root)

    generation_summary_path = _validate_file_spec(
        review.generation_summary, root
    )
    raw_dataset_path = _validate_file_spec(review.raw_dataset, root)
    canonical_dataset_path = _validate_file_spec(review.canonical_dataset, root)
    report = HeldoutGenerationReport.model_validate_json(
        generation_summary_path.read_text(encoding="utf-8")
    )
    if not report.valid:
        raise ValueError("held-out generation report is not valid")
    if report.dataset_sha256 != review.raw_dataset.sha256:
        raise ValueError("raw dataset hash disagrees with generation report")
    _validate_span_artifacts(report, generation_summary_path.parent)

    records = _load_jsonl_records(raw_dataset_path)
    decisions = {decision.case_id: decision for decision in review.decisions}
    case_ids = [_required_string(record, "id") for record in records]
    if len(records) != 20 or set(case_ids) != set(decisions):
        raise ValueError("manual review does not cover the raw held-out cases")

    finalized: list[dict[str, object]] = []
    corrected_case_ids: list[str] = []
    for record in records:
        case_id = _required_string(record, "id")
        query = _required_string(record, "query")
        answer = _required_string(record, "reference_answer")
        decision = decisions[case_id]
        if _text_sha256(query) != decision.original_query_sha256:
            raise ValueError(f"manual review query hash mismatch: {case_id}")
        if _text_sha256(answer) != decision.original_reference_answer_sha256:
            raise ValueError(f"manual review answer hash mismatch: {case_id}")

        updated = dict(record)
        if isinstance(decision, CorrectedReviewDecision):
            updated["query"] = decision.replacement_query
            updated["reference_answer"] = decision.replacement_reference_answer
            metadata = _required_record(updated, "metadata")
            updated["metadata"] = {
                **metadata,
                "manual_review": {
                    "status": decision.status,
                    "reason_code": decision.reason_code,
                    "note": decision.note,
                },
            }
            corrected_case_ids.append(case_id)
        finalized.append(updated)

    normalized_queries = [
        _required_string(record, "query").strip().casefold()
        for record in finalized
    ]
    canonical_queries = {
        case.query.strip().casefold() for case in load_cases(canonical_dataset_path)
    }
    canonical_duplicates = sorted(set(normalized_queries) & canonical_queries)
    if len(set(normalized_queries)) != 20:
        raise ValueError("finalized held-out queries are not unique")

    dataset_text = "".join(
        f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n"
        for record in finalized
    )
    atomic_write_text(dataset_path, dataset_text)
    loaded = load_cases(dataset_path)
    if len(loaded) != 20:
        raise ValueError("finalized held-out dataset did not round-trip")

    manifest = FinalizedHeldoutDatasetManifest(
        name="pre-v0.3-e4b-heldout-confirmation-dataset",
        generation_commit=report.git.commit,
        generation_summary=review.generation_summary,
        raw_dataset=review.raw_dataset,
        review_manifest=BaselineFileSpec(
            path=_display_path(review_source, root),
            sha256=sha256_file(review_source, "text_lf"),
            hash_mode="text_lf",
        ),
        canonical_dataset=review.canonical_dataset,
        dataset_path=_display_path(dataset_path, root),
        dataset_sha256=sha256_file(dataset_path, "text_lf"),
        canonical_query_duplicates=canonical_duplicates,
        manual_pass_count=sum(
            isinstance(decision, PassedReviewDecision)
            for decision in review.decisions
        ),
        manual_correction_count=len(corrected_case_ids),
        corrected_case_ids=corrected_case_ids,
        valid=not canonical_duplicates,
    )
    atomic_write_text(
        manifest_path,
        manifest.model_dump_json(indent=2) + "\n",
    )
    return manifest


def _validate_span_artifacts(
    report: HeldoutGenerationReport,
    generation_dir: Path,
) -> None:
    for span in report.resolved_spans:
        artifact_path = (generation_dir / span.artifact_path).resolve()
        if not artifact_path.is_relative_to(generation_dir.resolve()):
            raise ValueError("held-out span artifact escapes generation directory")
        if sha256_file(artifact_path, "text_lf") != span.artifact_sha256:
            raise ValueError(f"held-out span artifact hash mismatch: {span.span_id}")


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"held-out case {line_number} is not an object")
        records.append(cast(dict[str, object], payload))
    return records


def _required_string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"held-out case field is invalid: {key}")
    return value


def _required_record(record: dict[str, object], key: str) -> dict[str, object]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"held-out case field is not an object: {key}")
    return cast(dict[str, object], value)


def _validate_file_spec(spec: BaselineFileSpec, root: Path) -> Path:
    path = (root / spec.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"held-out reviewed input is invalid: {spec.path}")
    if sha256_file(path, spec.hash_mode) != spec.sha256:
        raise ValueError(f"held-out reviewed input hash mismatch: {spec.path}")
    return path


def _require_within_repository(path: Path, root: Path) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"held-out finalized path escapes repository: {path}")


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize the manually reviewed E4b held-out dataset."
    )
    parser.add_argument("--review", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument("--output-dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-manifest", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args(argv)
    try:
        root = Path.cwd().resolve()
        manifest = finalize_heldout_dataset(
            load_review_manifest(args.review),
            review_path=args.review,
            repository_root=root,
            output_dataset=args.output_dataset,
            output_manifest=args.output_manifest,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Held-out finalization failed: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset: {manifest.dataset_path}")
    print(f"Cases: {manifest.case_count}")
    print(f"Manual passes: {manifest.manual_pass_count}")
    print(f"Manual corrections: {manifest.manual_correction_count}")
    print(f"Canonical duplicates: {len(manifest.canonical_query_duplicates)}")
    print(f"Valid: {manifest.valid}")
    return 0 if manifest.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
