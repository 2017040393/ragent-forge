from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.generate_e4b_heldout import (
    _farthest_fill_indexes,
    _portable_span,
    _text_sha256,
    load_manifest,
)
from benchmarks.retrieval_baseline import sha256_file
from pydantic import ValidationError

from ragent_forge.app.services.evidence_span_service import EvidenceSpan


def _span(index: int, text: str = "evidence") -> EvidenceSpan:
    return EvidenceSpan(
        id=f"docs/source.pdf::span-{index:04d}",
        source_path="docs/source.pdf",
        document_id="docs/source.pdf",
        start_char=None,
        end_char=None,
        text=text,
        media_type="application/pdf",
        section_title=None,
        heading_path=(),
        block_types=("paragraph",),
        page_start=index + 1,
        page_end=index + 1,
        metadata={"source_path": "docs/source.pdf"},
    )


def test_checked_in_manifest_freezes_inputs_and_generation_contract() -> None:
    root = Path(__file__).parents[1]
    manifest = load_manifest()
    specs = [
        manifest.canonical_dataset,
        manifest.canonical_manifest,
        *manifest.sources,
    ]

    assert manifest.generation.case_count == 20
    assert manifest.generation.reasoning_effort == "medium"
    assert sum(len(source.selected_spans) for source in manifest.sources) == 10
    assert all(
        sha256_file(root / spec.path, spec.hash_mode) == spec.sha256
        for spec in specs
    )


def test_manifest_rejects_canonical_span_overlap() -> None:
    payload = load_manifest().model_dump(mode="json")
    sources = payload["sources"]
    assert isinstance(sources, list)
    first = sources[0]
    assert isinstance(first, dict)
    selected = first["selected_spans"]
    canonical = first["canonical_selected_span_indexes"]
    assert isinstance(selected, list)
    assert isinstance(canonical, list)
    first_span = selected[0]
    assert isinstance(first_span, dict)
    first_span["index"] = canonical[0]

    with pytest.raises(ValidationError, match="overlap canonical"):
        type(load_manifest()).model_validate(payload)


def test_farthest_fill_is_deterministic_and_prefers_lower_index_ties() -> None:
    spans = [_span(index) for index in range(11)]

    selected = _farthest_fill_indexes(spans, occupied=[2, 8], count=3)

    assert selected == [5, 0, 10]


def test_portable_span_rewrites_identity_without_changing_text() -> None:
    original = replace(
        _span(7, "quotient evidence"),
        source_path="C:\\repo\\docs\\source.pdf",
        document_id="C:\\repo\\docs\\source.pdf",
    )

    portable = _portable_span(original, "docs/source.pdf", 7)

    assert portable.id == "docs/source.pdf::span-0007"
    assert portable.source_path == "docs/source.pdf"
    assert portable.document_id == "docs/source.pdf"
    assert portable.text == original.text
    assert portable.metadata["text_sha256"] == _text_sha256(original.text)
