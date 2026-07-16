from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, Field, model_validator

from benchmarks.retrieval_baseline import collect_git_state, sha256_file
from ragent_forge.app.models import AppConfig
from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.eval_dataset_generation_service import (
    Difficulty,
    EvalDatasetGenerationService,
    GeneratedEvalCase,
    QuestionType,
    TextGenerationClient,
)
from ragent_forge.app.services.evaluation.baseline import (
    BaselineFileSpec,
    BaselineGitState,
)
from ragent_forge.app.services.evaluation.cases import load_cases
from ragent_forge.app.services.evidence_span_service import (
    EvidenceSpan,
    EvidenceSpanService,
)
from ragent_forge.composition import build_text_generation_client
from ragent_forge.infrastructure.eval_output import write_generated_eval_jsonl
from ragent_forge.infrastructure.local_workspace import LocalWorkspace
from ragent_forge.infrastructure.storage import atomic_write_text

DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "generate_e4b_heldout_manifest.json"
)


class HeldoutSpanSpec(BaseModel):
    index: int = Field(ge=0)
    page: int = Field(gt=0)
    text_chars: int = Field(ge=1000, le=1200)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HeldoutSourceSpec(BaselineFileSpec):
    canonical_selected_span_indexes: list[int] = Field(min_length=1)
    selected_spans: list[HeldoutSpanSpec] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def _unique_span_indexes(self) -> Self:
        selected = [span.index for span in self.selected_spans]
        if len(selected) != len(set(selected)):
            raise ValueError("held-out span indexes must be unique")
        if set(selected) & set(self.canonical_selected_span_indexes):
            raise ValueError("held-out spans overlap canonical spans")
        return self


class HeldoutGenerationSpec(BaseModel):
    provider: Literal["openai_responses"] = "openai_responses"
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: Literal["medium"] = "medium"
    temperature: float = Field(default=0.2, ge=0, le=2)
    questions_per_span: Literal[2] = 2
    case_count: Literal[20] = 20
    case_id_prefix: Literal["e4b-heldout-"] = "e4b-heldout-"

    @model_validator(mode="after")
    def _fixed_temperature(self) -> Self:
        if self.temperature != 0.2:
            raise ValueError("held-out generation temperature must remain 0.2")
        return self


class HeldoutGenerationManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    canonical_dataset: BaselineFileSpec
    canonical_manifest: BaselineFileSpec
    source_root: str = Field(min_length=1)
    minimum_eligible_span_chars: Literal[1000] = 1000
    sources: list[HeldoutSourceSpec] = Field(min_length=2, max_length=2)
    generation: HeldoutGenerationSpec

    @model_validator(mode="after")
    def _fixed_sources(self) -> Self:
        paths = [source.path for source in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("held-out sources must be unique")
        if sum(len(source.selected_spans) for source in self.sources) != 10:
            raise ValueError("held-out generation requires exactly 10 spans")
        return self


class HeldoutGeneratedItem(BaseModel):
    query: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    question_type: QuestionType
    difficulty: Difficulty


class HeldoutSpanArtifact(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_path: str
    span_id: str
    span_index: int = Field(ge=0)
    page: int = Field(gt=0)
    text_chars: int = Field(ge=1000, le=1200)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: HeldoutGenerationSpec
    items: list[HeldoutGeneratedItem] = Field(min_length=2, max_length=2)


class HeldoutResolvedSpan(BaseModel):
    source_path: str
    span_id: str
    span_index: int
    page: int
    text_chars: int
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_path: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HeldoutGenerationReport(BaseModel):
    schema_version: Literal[1] = 1
    benchmark: str
    description: str
    generated_at: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git: BaselineGitState
    generation: HeldoutGenerationSpec
    resolved_spans: list[HeldoutResolvedSpan] = Field(min_length=10, max_length=10)
    dataset_path: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: Literal[20] = 20
    unique_query_count: int = Field(ge=20, le=20)
    canonical_query_duplicates: list[str]
    valid: bool


class GenerationClientFactory(Protocol):
    def __call__(self, config: AppConfig) -> TextGenerationClient: ...


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> HeldoutGenerationManifest:
    return HeldoutGenerationManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def generate_heldout_dataset(
    manifest: HeldoutGenerationManifest,
    *,
    manifest_path: str | Path,
    repository_root: str | Path,
    workspace: LocalWorkspace,
    output_dir: str | Path,
    git_state: BaselineGitState,
    resume: bool = False,
    client_factory: GenerationClientFactory = build_text_generation_client,
) -> HeldoutGenerationReport:
    root = Path(repository_root).resolve()
    source_manifest_path = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and not resume:
        raise FileExistsError(f"held-out output directory exists: {destination}")
    if resume and not destination.is_dir():
        raise FileNotFoundError(f"held-out resume directory is missing: {destination}")

    canonical_queries = _validate_frozen_inputs(manifest, root)
    spans = resolve_selected_spans(manifest, root)
    config = _generation_config(manifest, ConfigService(workspace).load())
    if resume:
        _validate_resume_manifest(destination, manifest)
    else:
        destination.mkdir(parents=True)
        (destination / "span-runs").mkdir()
        atomic_write_text(
            destination / "manifest.json",
            manifest.model_dump_json(indent=2) + "\n",
        )

    client: TextGenerationClient | None = None
    artifacts: list[tuple[Path, HeldoutSpanArtifact]] = []
    for position, span in enumerate(spans, start=1):
        source_path = _repository_relative_source(span, root)
        span_index = _span_index(span.id)
        relative_path = Path("span-runs") / (
            f"{position:02d}-{_safe_source_stem(source_path)}-span-{span_index:04d}.json"
        )
        artifact_path = destination / relative_path
        if resume and artifact_path.is_file():
            artifact = HeldoutSpanArtifact.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
            _validate_span_artifact(
                artifact,
                manifest=manifest,
                span=span,
                source_path=source_path,
                git_commit=git_state.commit,
            )
        else:
            if client is None:
                client = client_factory(config)
            artifact = _generate_span_artifact(
                manifest,
                span=span,
                source_path=source_path,
                git_commit=git_state.commit,
                client=client,
            )
            atomic_write_text(
                artifact_path,
                artifact.model_dump_json(indent=2) + "\n",
            )
        artifacts.append((relative_path, artifact))

    generated_cases = _assemble_cases(manifest, spans, artifacts)
    queries = [case.query.strip().casefold() for case in generated_cases]
    duplicates = sorted(set(queries) & canonical_queries)
    if len(set(queries)) != manifest.generation.case_count:
        raise ValueError("held-out generated queries are not unique")
    dataset_path = destination / "cases.jsonl"
    write_generated_eval_jsonl(generated_cases, dataset_path, overwrite=resume)
    loaded = load_cases(dataset_path)
    if len(loaded) != manifest.generation.case_count:
        raise ValueError("held-out dataset did not round-trip through the case loader")

    resolved_spans = [
        HeldoutResolvedSpan(
            source_path=artifact.source_path,
            span_id=artifact.span_id,
            span_index=artifact.span_index,
            page=artifact.page,
            text_chars=artifact.text_chars,
            text_sha256=artifact.text_sha256,
            artifact_path=relative_path.as_posix(),
            artifact_sha256=sha256_file(destination / relative_path, "text_lf"),
        )
        for relative_path, artifact in artifacts
    ]
    report = HeldoutGenerationReport(
        benchmark=manifest.name,
        description=manifest.description,
        generated_at=datetime.now(UTC).isoformat(),
        manifest_path=_display_path(source_manifest_path, root),
        manifest_sha256=sha256_file(source_manifest_path, "text_lf"),
        git=git_state,
        generation=manifest.generation,
        resolved_spans=resolved_spans,
        dataset_path="cases.jsonl",
        dataset_sha256=sha256_file(dataset_path, "text_lf"),
        unique_query_count=len(set(queries)),
        canonical_query_duplicates=duplicates,
        valid=(not git_state.dirty and not duplicates),
    )
    atomic_write_text(
        destination / "summary.json",
        report.model_dump_json(indent=2) + "\n",
    )
    return report


def resolve_selected_spans(
    manifest: HeldoutGenerationManifest,
    repository_root: Path,
) -> list[EvidenceSpan]:
    source_root = (repository_root / manifest.source_root).resolve()
    extracted = EvidenceSpanService(
        min_chars=250,
        max_chars=1200,
        include_pdf=True,
    ).extract(source_root)
    by_source: dict[str, list[EvidenceSpan]] = {}
    for span in extracted:
        if span.media_type != "application/pdf":
            continue
        relative = _repository_relative_source(span, repository_root)
        by_source.setdefault(relative, []).append(span)

    selected: list[EvidenceSpan] = []
    for source_spec in manifest.sources:
        eligible = [
            span
            for span in by_source.get(source_spec.path, [])
            if len(span.text) >= manifest.minimum_eligible_span_chars
        ]
        expected_indexes = [span.index for span in source_spec.selected_spans]
        computed = _farthest_fill_indexes(
            eligible,
            occupied=source_spec.canonical_selected_span_indexes,
            count=5,
        )
        if computed != expected_indexes:
            raise ValueError(f"held-out farthest-fill drifted: {source_spec.path}")
        eligible_by_index = {_span_index(span.id): span for span in eligible}
        for span_spec in source_spec.selected_spans:
            span = eligible_by_index.get(span_spec.index)
            if span is None:
                raise ValueError(
                    "held-out span is missing: "
                    f"{source_spec.path}:{span_spec.index}"
                )
            digest = _text_sha256(span.text)
            if (
                span.page_start != span_spec.page
                or len(span.text) != span_spec.text_chars
                or digest != span_spec.text_sha256
            ):
                raise ValueError(
                    "held-out span provenance mismatch: "
                    f"{source_spec.path}:{span_spec.index}"
                )
            selected.append(_portable_span(span, source_spec.path, span_spec.index))
    if len(selected) != 10:
        raise ValueError("held-out selection did not resolve 10 spans")
    return selected


def _farthest_fill_indexes(
    spans: Sequence[EvidenceSpan],
    *,
    occupied: Sequence[int],
    count: int,
) -> list[int]:
    available = {_span_index(span.id) for span in spans} - set(occupied)
    chosen: list[int] = []
    for _ in range(count):
        reference = [*occupied, *chosen]
        if not available or not reference:
            raise ValueError("held-out farthest-fill has insufficient span indexes")
        selected = max(
            available,
            key=lambda index: (
                min(abs(index - existing) for existing in reference),
                -index,
            ),
        )
        chosen.append(selected)
        available.remove(selected)
    return chosen


def _generate_span_artifact(
    manifest: HeldoutGenerationManifest,
    *,
    span: EvidenceSpan,
    source_path: str,
    git_commit: str,
    client: TextGenerationClient,
) -> HeldoutSpanArtifact:
    report = EvalDatasetGenerationService(
        generator=client,
        questions_per_span=manifest.generation.questions_per_span,
    ).generate([span], max_cases=manifest.generation.questions_per_span)
    if report.errors or len(report.cases) != manifest.generation.questions_per_span:
        raise RuntimeError(
            f"held-out generation failed for {span.id}: {report.errors}"
        )
    items = [
        HeldoutGeneratedItem(
            query=case.query,
            reference_answer=case.reference_answer,
            question_type=case.question_type,
            difficulty=case.difficulty,
        )
        for case in report.cases
    ]
    return HeldoutSpanArtifact(
        benchmark=manifest.name,
        git_commit=git_commit,
        source_path=source_path,
        span_id=span.id,
        span_index=_span_index(span.id),
        page=_required_page(span),
        text_chars=len(span.text),
        text_sha256=_text_sha256(span.text),
        generation=manifest.generation,
        items=items,
    )


def _assemble_cases(
    manifest: HeldoutGenerationManifest,
    spans: Sequence[EvidenceSpan],
    artifacts: Sequence[tuple[Path, HeldoutSpanArtifact]],
) -> list[GeneratedEvalCase]:
    cases: list[GeneratedEvalCase] = []
    for span, (_, artifact) in zip(spans, artifacts, strict=True):
        for item in artifact.items:
            case_number = len(cases) + 1
            cases.append(
                GeneratedEvalCase(
                    id=f"{manifest.generation.case_id_prefix}{case_number:06d}",
                    query=item.query,
                    evidence_spans=[span],
                    reference_answer=item.reference_answer,
                    question_type=item.question_type,
                    difficulty=item.difficulty,
                    generation_method="llm_synthetic_span_e4b_heldout_v1",
                    metadata={
                        "heldout_protocol": manifest.name,
                        "source_span_index": artifact.span_index,
                        "generation_provider": manifest.generation.provider,
                        "generation_model": manifest.generation.model,
                        "generation_reasoning_effort": (
                            manifest.generation.reasoning_effort
                        ),
                        "generation_temperature": manifest.generation.temperature,
                    },
                )
            )
    return cases


def _validate_span_artifact(
    artifact: HeldoutSpanArtifact,
    *,
    manifest: HeldoutGenerationManifest,
    span: EvidenceSpan,
    source_path: str,
    git_commit: str,
) -> None:
    if (
        artifact.benchmark != manifest.name
        or artifact.git_commit != git_commit
        or artifact.source_path != source_path
        or artifact.span_id != span.id
        or artifact.span_index != _span_index(span.id)
        or artifact.page != _required_page(span)
        or artifact.text_chars != len(span.text)
        or artifact.text_sha256 != _text_sha256(span.text)
        or artifact.generation != manifest.generation
    ):
        raise ValueError(f"held-out resume artifact mismatch: {artifact.span_id}")


def _validate_frozen_inputs(
    manifest: HeldoutGenerationManifest,
    repository_root: Path,
) -> set[str]:
    specs = [
        manifest.canonical_dataset,
        manifest.canonical_manifest,
        *manifest.sources,
    ]
    for spec in specs:
        path = _resolve_file(repository_root, spec.path)
        if sha256_file(path, spec.hash_mode) != spec.sha256:
            raise ValueError(f"held-out frozen input hash mismatch: {spec.path}")
    canonical = load_cases(repository_root / manifest.canonical_dataset.path)
    return {case.query.strip().casefold() for case in canonical}


def _generation_config(
    manifest: HeldoutGenerationManifest,
    config: AppConfig,
) -> AppConfig:
    generation = config.generation
    if (
        generation.provider != manifest.generation.provider
        or generation.model != manifest.generation.model
        or (generation.base_url or "").rstrip("/")
        != manifest.generation.base_url.rstrip("/")
    ):
        raise ValueError("held-out generation config does not match the manifest")
    return config.model_copy(
        update={
            "generation": generation.model_copy(
                update={
                    "temperature": manifest.generation.temperature,
                    "reasoning_effort": manifest.generation.reasoning_effort,
                }
            )
        }
    )


def _portable_span(
    span: EvidenceSpan,
    source_path: str,
    span_index: int,
) -> EvidenceSpan:
    metadata = dict(span.metadata)
    metadata.update(
        {
            "source_path": source_path,
            "document_id": source_path,
            "text_sha256": _text_sha256(span.text),
        }
    )
    return replace(
        span,
        id=f"{source_path}::span-{span_index:04d}",
        source_path=source_path,
        document_id=source_path,
        metadata=metadata,
    )


def _repository_relative_source(span: EvidenceSpan, repository_root: Path) -> str:
    source = Path(span.source_path).resolve()
    try:
        return source.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"held-out source escapes repository: {source}") from exc


def _validate_resume_manifest(
    output_dir: Path,
    manifest: HeldoutGenerationManifest,
) -> None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"held-out resume manifest is missing: {path}")
    existing = HeldoutGenerationManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if existing != manifest:
        raise ValueError("held-out resume manifest mismatch")


def _span_index(span_id: str) -> int:
    marker = "::span-"
    if marker not in span_id:
        raise ValueError(f"invalid evidence span ID: {span_id}")
    value = span_id.rsplit(marker, 1)[1]
    if not value.isdigit():
        raise ValueError(f"invalid evidence span index: {span_id}")
    return int(value)


def _required_page(span: EvidenceSpan) -> int:
    if span.page_start is None:
        raise ValueError(f"held-out PDF span has no page: {span.id}")
    return span.page_start


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_source_stem(source_path: str) -> str:
    stem = Path(source_path).stem.lower()
    normalized = "".join(
        character if character.isalnum() else "-" for character in stem
    )
    return normalized.strip("-")[:32]


def _resolve_file(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to(repository_root) or not path.is_file():
        raise FileNotFoundError(f"held-out input file is invalid: {value}")
    return path


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _print_summary(report: HeldoutGenerationReport, output_dir: Path) -> None:
    print(f"Held-out dataset: {report.benchmark}")
    print(f"Cases: {report.case_count}")
    print(f"Unique queries: {report.unique_query_count}")
    print(f"Canonical duplicates: {len(report.canonical_query_duplicates)}")
    print(f"Dataset: {output_dir / report.dataset_path}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Valid: {report.valid}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the frozen E4b held-out retrieval dataset."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = Path.cwd().resolve()
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(completed.stdout.strip()).resolve()
        git_state = collect_git_state(root)
        if git_state.dirty:
            raise ValueError("held-out dataset generation requires a clean Git tree")
        report = generate_heldout_dataset(
            load_manifest(args.manifest),
            manifest_path=args.manifest,
            repository_root=root,
            workspace=LocalWorkspace(args.workspace),
            output_dir=args.output_dir,
            git_state=git_state,
            resume=args.resume,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Held-out generation failed: {exc}", file=sys.stderr)
        return 1
    _print_summary(report, Path(args.output_dir).resolve())
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
