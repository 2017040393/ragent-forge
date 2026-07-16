from __future__ import annotations

import math

import pytest

from ragent_forge.core.retrieval.context_fragments import (
    RankedFragmentCandidate,
    allocate_fragment_chars,
    build_evidence_window_scorer,
    build_query_window_scorer,
    fragment_windows,
    fragments_are_traceable,
    normalized_token_ngrams,
    render_fragments,
    select_ranked_fragments,
)


def _candidates(*, text: str | None = None) -> list[RankedFragmentCandidate]:
    return [
        RankedFragmentCandidate(
            rank=rank,
            chunk_id=f"chunk-{rank}",
            source_path="docs/source.pdf",
            source_label="source.pdf",
            page_label=str(rank),
            text=text or (f"candidate {rank} " + ("body " * 180)),
            signal_text="formula quotient vector space",
        )
        for rank in range(1, 6)
    ]


def test_allocations_include_render_overhead_and_water_fill_short_chunks() -> None:
    candidates = _candidates()
    candidates[0] = RankedFragmentCandidate(
        rank=1,
        chunk_id="chunk-1",
        source_path="docs/source.pdf",
        source_label="source.pdf",
        page_label="1",
        text="short",
    )

    allocations = allocate_fragment_chars(
        candidates,
        max_context_chars=3072,
        max_fragment_chars=640,
    )

    assert allocations[0] == 5
    assert all(allocation <= 640 for allocation in allocations)
    assert allocations[1:] == sorted(allocations[1:], reverse=True)


def test_query_selector_keeps_all_ranks_and_chooses_relevant_window() -> None:
    text = (
        ("unrelated material. " * 35)
        + "The quotient V slash U forms a vector space when U is a subspace. "
        + ("trailing material. " * 35)
    )
    candidates = _candidates(text=text)
    chunk_text_by_id = {candidate.chunk_id: candidate.text for candidate in candidates}

    fragments = select_ranked_fragments(
        candidates,
        max_context_chars=3072,
        max_fragment_chars=640,
        stride_chars=112,
        scorer=build_query_window_scorer(
            "Under what condition does the quotient V/U form a vector space?"
        ),
    )

    assert [fragment.rank for fragment in fragments] == [1, 2, 3, 4, 5]
    assert [fragment.chunk_id for fragment in fragments] == [
        candidate.chunk_id for candidate in candidates
    ]
    assert all("quotient" in fragment.text for fragment in fragments)
    assert fragments_are_traceable(fragments, chunk_text_by_id)
    assert math.ceil(len(render_fragments(fragments)) / 4) <= 768


def test_oracle_scorer_selects_window_with_gold_evidence() -> None:
    candidates = _candidates(
        text=("noise " * 120) + "stable dimension preserves geometry" + (" tail" * 120)
    )
    fragments = select_ranked_fragments(
        candidates,
        max_context_chars=3072,
        max_fragment_chars=640,
        stride_chars=112,
        scorer=build_evidence_window_scorer(
            "Below the stable dimension the projection does not preserve geometry.",
            ngram_size=3,
        ),
    )

    assert all("stable dimension preserves geometry" in item.text for item in fragments)
    assert normalized_token_ngrams(
        fragments[0].text,
        ngram_size=3,
    ) & normalized_token_ngrams(
        "stable dimension preserves geometry",
        ngram_size=3,
    )


def test_fragment_windows_cover_start_and_end_of_long_unit() -> None:
    text = "x" * 1000
    windows = fragment_windows(text, max_chars=448, stride_chars=112)

    assert windows[0][0] == 0
    assert any(end == len(text) for _, end in windows)
    assert all(0 < end - start <= 448 for start, end in windows)


def test_selector_rejects_noncontiguous_ranks() -> None:
    candidates = _candidates()
    candidates[1] = RankedFragmentCandidate(
        rank=3,
        chunk_id="different",
        source_path="docs/source.pdf",
        source_label="source.pdf",
        page_label="3",
        text="text",
    )

    with pytest.raises(ValueError, match="ranks must be contiguous"):
        select_ranked_fragments(
            candidates,
            max_context_chars=3072,
            max_fragment_chars=640,
            stride_chars=112,
            scorer=build_query_window_scorer("query"),
        )
