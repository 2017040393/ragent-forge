from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_BOUNDARY_RE = re.compile(r"(?:\r?\n)+|(?<=[.!?;:])\s+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "under",
        "what",
        "when",
        "which",
        "with",
    }
)


@dataclass(frozen=True)
class RankedFragmentCandidate:
    rank: int
    chunk_id: str
    source_path: str
    source_label: str
    page_label: str
    text: str
    signal_text: str = ""


@dataclass(frozen=True)
class FragmentScoreComponents:
    unique_query_tokens: int
    query_token_occurrences: int
    query_bigram_coverage: int
    signal_token_coverage: int


@dataclass(frozen=True)
class ContextFragment:
    rank: int
    chunk_id: str
    source_path: str
    source_label: str
    page_label: str
    start_char: int
    end_char: int
    text: str
    truncated_left: bool
    truncated_right: bool
    score: FragmentScoreComponents

    @property
    def header(self) -> str:
        return fragment_header(
            rank=self.rank,
            source_label=self.source_label,
            page_label=self.page_label,
        )


FragmentWindowScorer = Callable[
    [RankedFragmentCandidate, str],
    FragmentScoreComponents,
]


def select_ranked_fragments(
    candidates: Sequence[RankedFragmentCandidate],
    *,
    max_context_chars: int,
    max_fragment_chars: int,
    stride_chars: int,
    scorer: FragmentWindowScorer,
) -> list[ContextFragment]:
    """Select one traceable contiguous fragment from every ranked candidate."""

    _validate_fragment_inputs(
        candidates,
        max_context_chars=max_context_chars,
        max_fragment_chars=max_fragment_chars,
        stride_chars=stride_chars,
    )
    allocations = allocate_fragment_chars(
        candidates,
        max_context_chars=max_context_chars,
        max_fragment_chars=max_fragment_chars,
    )
    fragments: list[ContextFragment] = []
    for candidate, allocation in zip(candidates, allocations, strict=True):
        windows = fragment_windows(
            candidate.text,
            max_chars=allocation,
            stride_chars=stride_chars,
        )
        if not windows:
            raise ValueError(f"fragment candidate has no window: {candidate.chunk_id}")
        start, end = max(
            windows,
            key=lambda window: _window_key(
                candidate,
                window,
                scorer=scorer,
            ),
        )
        text = candidate.text[start:end]
        fragments.append(
            ContextFragment(
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
                score=scorer(candidate, text),
            )
        )
    rendered = render_fragments(fragments)
    if len(rendered) > max_context_chars:
        raise ValueError("rendered fragments exceed the context budget")
    return fragments


def allocate_fragment_chars(
    candidates: Sequence[RankedFragmentCandidate],
    *,
    max_context_chars: int,
    max_fragment_chars: int,
) -> list[int]:
    if not candidates:
        return []
    headers = [
        fragment_header(
            rank=candidate.rank,
            source_label=candidate.source_label,
            page_label=candidate.page_label,
        )
        for candidate in candidates
    ]
    separator_chars = len("\n\n") * (len(candidates) - 1)
    available = max_context_chars - sum(len(header) + 1 for header in headers)
    available -= separator_chars
    if available < len(candidates):
        raise ValueError("context budget cannot represent every ranked candidate")

    capacities = [
        min(len(candidate.text), max_fragment_chars) for candidate in candidates
    ]
    if any(capacity < 1 for capacity in capacities):
        raise ValueError("fragment candidates must contain non-empty text")
    allocations = [0] * len(candidates)
    active = list(range(len(candidates)))
    remaining = available
    while remaining > 0 and active:
        share, remainder = divmod(remaining, len(active))
        if share == 0:
            share = 1
            remainder = 0
        granted = 0
        next_active: list[int] = []
        for position, index in enumerate(active):
            requested = share + (1 if position < remainder else 0)
            capacity = capacities[index] - allocations[index]
            grant = min(requested, capacity, remaining - granted)
            allocations[index] += grant
            granted += grant
            if allocations[index] < capacities[index]:
                next_active.append(index)
            if granted == remaining:
                break
        if granted == 0:
            break
        remaining -= granted
        active = next_active
    if any(allocation < 1 for allocation in allocations):
        raise ValueError("context budget produced an empty candidate allocation")
    return allocations


def fragment_windows(
    text: str,
    *,
    max_chars: int,
    stride_chars: int,
) -> list[tuple[int, int]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if stride_chars <= 0:
        raise ValueError("stride_chars must be greater than 0")
    if not text:
        return []
    if len(text) <= max_chars:
        return [(0, len(text))]

    boundaries = {0, len(text)}
    boundaries.update(match.end() for match in _BOUNDARY_RE.finditer(text))
    ordered_boundaries = sorted(boundaries)
    starts = set(ordered_boundaries[:-1])
    for unit_start, unit_end in pairwise(ordered_boundaries):
        if unit_end - unit_start > max_chars:
            starts.update(range(unit_start, unit_end, stride_chars))
    starts.add(max(0, len(text) - max_chars))

    windows: set[tuple[int, int]] = set()
    minimum_preferred_length = max(1, max_chars // 2)
    for raw_start in sorted(starts):
        start = _trim_left(text, raw_start, len(text))
        maximum_end = min(len(text), start + max_chars)
        preferred_ends = [
            boundary
            for boundary in ordered_boundaries
            if start + minimum_preferred_length <= boundary <= maximum_end
        ]
        end = max(preferred_ends, default=maximum_end)
        end = _trim_right(text, start, end)
        if end > start:
            windows.add((start, end))
    return sorted(windows)


def build_query_window_scorer(query: str) -> FragmentWindowScorer:
    query_tokens = normalized_tokens(query)
    content_tokens = [token for token in query_tokens if token not in _STOPWORDS]
    effective_tokens = content_tokens or query_tokens
    query_token_set = set(effective_tokens)
    query_bigrams = set(_token_ngrams(effective_tokens, 2))

    def score(
        candidate: RankedFragmentCandidate,
        text: str,
    ) -> FragmentScoreComponents:
        window_tokens = normalized_tokens(text)
        window_token_set = set(window_tokens)
        signal_tokens = set(normalized_tokens(candidate.signal_text))
        return FragmentScoreComponents(
            unique_query_tokens=len(query_token_set & window_token_set),
            query_token_occurrences=sum(
                token in query_token_set for token in window_tokens
            ),
            query_bigram_coverage=len(
                query_bigrams & set(_token_ngrams(window_tokens, 2))
            ),
            signal_token_coverage=len(
                query_token_set & window_token_set & signal_tokens
            ),
        )

    return score


def build_evidence_window_scorer(
    evidence_text: str,
    *,
    ngram_size: int,
) -> FragmentWindowScorer:
    evidence_tokens = normalized_tokens(evidence_text)
    evidence_token_set = set(evidence_tokens)
    evidence_bigrams = set(_token_ngrams(evidence_tokens, 2))
    evidence_ngrams = set(_token_ngrams(evidence_tokens, ngram_size))

    def score(
        candidate: RankedFragmentCandidate,
        text: str,
    ) -> FragmentScoreComponents:
        del candidate
        window_tokens = normalized_tokens(text)
        window_ngrams = set(_token_ngrams(window_tokens, ngram_size))
        return FragmentScoreComponents(
            unique_query_tokens=len(evidence_ngrams & window_ngrams),
            query_token_occurrences=sum(
                token in evidence_token_set for token in window_tokens
            ),
            query_bigram_coverage=len(
                evidence_bigrams & set(_token_ngrams(window_tokens, 2))
            ),
            signal_token_coverage=0,
        )

    return score


def normalized_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_RE.findall(normalized)


def normalized_token_ngrams(
    text: str,
    *,
    ngram_size: int,
) -> set[tuple[str, ...]]:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be greater than 0")
    return set(_token_ngrams(normalized_tokens(text), ngram_size))


def fragment_header(*, rank: int, source_label: str, page_label: str) -> str:
    return f"[rank={rank} source={source_label} page={page_label}]"


def render_fragments(fragments: Sequence[ContextFragment]) -> str:
    return "\n\n".join(f"{fragment.header}\n{fragment.text}" for fragment in fragments)


def fragments_are_traceable(
    fragments: Sequence[ContextFragment],
    chunk_text_by_id: Mapping[str, str],
) -> bool:
    for fragment in fragments:
        chunk_text = chunk_text_by_id.get(fragment.chunk_id)
        if chunk_text is None:
            return False
        if not 0 <= fragment.start_char < fragment.end_char <= len(chunk_text):
            return False
        if chunk_text[fragment.start_char : fragment.end_char] != fragment.text:
            return False
    return True


def _window_key(
    candidate: RankedFragmentCandidate,
    window: tuple[int, int],
    *,
    scorer: FragmentWindowScorer,
) -> tuple[int, int, int, int, int, int]:
    start, end = window
    score = scorer(candidate, candidate.text[start:end])
    return (
        score.unique_query_tokens,
        score.query_token_occurrences,
        score.query_bigram_coverage,
        score.signal_token_coverage,
        end - start,
        -start,
    )


def _validate_fragment_inputs(
    candidates: Sequence[RankedFragmentCandidate],
    *,
    max_context_chars: int,
    max_fragment_chars: int,
    stride_chars: int,
) -> None:
    if not candidates:
        raise ValueError("fragment selection requires ranked candidates")
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than 0")
    if max_fragment_chars <= 0:
        raise ValueError("max_fragment_chars must be greater than 0")
    if stride_chars <= 0:
        raise ValueError("stride_chars must be greater than 0")
    ranks = [candidate.rank for candidate in candidates]
    if ranks != list(range(1, len(candidates) + 1)):
        raise ValueError("fragment candidate ranks must be contiguous")
    chunk_ids = [candidate.chunk_id for candidate in candidates]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("fragment candidate chunk IDs must be unique")


def _token_ngrams(
    tokens: Sequence[str],
    size: int,
) -> list[tuple[str, ...]]:
    if size <= 0:
        raise ValueError("ngram size must be greater than 0")
    return [
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    ]


def _trim_left(text: str, start: int, end: int) -> int:
    while start < end and text[start].isspace():
        start += 1
    return start


def _trim_right(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end
