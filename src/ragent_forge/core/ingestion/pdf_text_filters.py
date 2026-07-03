from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ragent_forge.core.ingestion.pdf_reading_order import PdfTextLine


@dataclass(frozen=True)
class TableDedupResult:
    text: str
    removed_lines: int
    strategy: str = "cell_text_line_filter"


@dataclass(frozen=True)
class HeaderFooterPageResult:
    text: str
    header_removed_lines: int
    footer_removed_lines: int
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class HeaderFooterFilterResult:
    pages: tuple[HeaderFooterPageResult, ...]
    suspected_headers_filtered: int
    suspected_footers_filtered: int


@dataclass(frozen=True)
class _HeaderFooterCandidate:
    text: str
    normalized: str
    position: Literal["header", "footer"]


def remove_table_text_duplicates(
    text: str,
    tables: Sequence[Sequence[Sequence[Any] | None]],
) -> TableDedupResult:
    table_tokens, row_tokens = _table_token_sets(tables)
    if not table_tokens:
        return TableDedupResult(text=text, removed_lines=0)

    kept_lines: list[str] = []
    removed_lines = 0
    for line in text.splitlines():
        tokens = _tokens(line)
        if _is_table_duplicate(tokens, table_tokens, row_tokens):
            removed_lines += 1
        else:
            kept_lines.append(line)
    return TableDedupResult(
        text="\n".join(kept_lines).strip(),
        removed_lines=removed_lines,
    )


def filter_repeated_header_footer_lines(
    pages: Sequence[Sequence[PdfTextLine]],
) -> HeaderFooterFilterResult:
    candidates = _detect_header_footer_candidates(pages)
    if not candidates:
        return HeaderFooterFilterResult(
            pages=tuple(
                HeaderFooterPageResult(
                    text="\n".join(line.text for line in page).strip(),
                    header_removed_lines=0,
                    footer_removed_lines=0,
                    candidates=(),
                )
                for page in pages
            ),
            suspected_headers_filtered=0,
            suspected_footers_filtered=0,
        )

    candidate_by_normalized = {
        candidate.normalized: candidate for candidate in candidates
    }
    page_results: list[HeaderFooterPageResult] = []
    total_headers = 0
    total_footers = 0
    for page in pages:
        kept_lines: list[str] = []
        header_removed = 0
        footer_removed = 0
        used_candidates: list[str] = []
        for line in page:
            normalized = _normalize_line(line.text)
            candidate = candidate_by_normalized.get(normalized)
            if candidate is None:
                kept_lines.append(line.text)
                continue
            used_candidates.append(candidate.text)
            if candidate.position == "header":
                header_removed += 1
            else:
                footer_removed += 1
        total_headers += header_removed
        total_footers += footer_removed
        page_results.append(
            HeaderFooterPageResult(
                text="\n".join(kept_lines).strip(),
                header_removed_lines=header_removed,
                footer_removed_lines=footer_removed,
                candidates=tuple(_unique(used_candidates)),
            )
        )

    return HeaderFooterFilterResult(
        pages=tuple(page_results),
        suspected_headers_filtered=total_headers,
        suspected_footers_filtered=total_footers,
    )


def _detect_header_footer_candidates(
    pages: Sequence[Sequence[PdfTextLine]],
) -> tuple[_HeaderFooterCandidate, ...]:
    if len(pages) < 2:
        return ()

    occurrences: dict[str, list[tuple[str, Literal["header", "footer"]]]] = (
        defaultdict(list)
    )
    for page in pages:
        for index, line in enumerate(page):
            position = _line_position(index, len(page))
            if position is None or not _is_safe_header_footer_line(line.text):
                continue
            occurrences[_normalize_line(line.text)].append((line.text, position))

    candidates: list[_HeaderFooterCandidate] = []
    for normalized, values in occurrences.items():
        if len(values) < 2:
            continue
        positions = [position for _, position in values]
        position = Counter(positions).most_common(1)[0][0]
        text = values[0][0]
        candidates.append(
            _HeaderFooterCandidate(
                text=text,
                normalized=normalized,
                position=position,
            )
        )
    return tuple(candidates)


def _line_position(
    index: int,
    line_count: int,
) -> Literal["header", "footer"] | None:
    if index < 2:
        return "header"
    if index >= max(line_count - 2, 0):
        return "footer"
    return None


def _is_safe_header_footer_line(text: str) -> bool:
    normalized = _normalize_line(text)
    if not normalized or len(normalized) > 80:
        return False
    token_count = len(_tokens(normalized))
    return 1 <= token_count <= 8


def _table_token_sets(
    tables: Sequence[Sequence[Sequence[Any] | None]],
) -> tuple[set[str], list[set[str]]]:
    table_tokens: set[str] = set()
    row_tokens: list[set[str]] = []
    for table in tables:
        for row in table:
            if row is None:
                continue
            normalized_row_tokens: set[str] = set()
            for cell in row:
                cell_tokens = _tokens("" if cell is None else str(cell))
                table_tokens.update(cell_tokens)
                normalized_row_tokens.update(cell_tokens)
            if normalized_row_tokens:
                row_tokens.append(normalized_row_tokens)
    return table_tokens, row_tokens


def _is_table_duplicate(
    line_tokens: list[str],
    table_tokens: set[str],
    row_tokens: list[set[str]],
) -> bool:
    if len(line_tokens) < 2:
        return False
    line_token_set = set(line_tokens)
    table_overlap = len(line_token_set & table_tokens)
    if table_overlap >= 2 and table_overlap / len(line_token_set) >= 0.8:
        return True
    for row_token_set in row_tokens:
        if len(row_token_set) < 2:
            continue
        row_overlap = len(line_token_set & row_token_set)
        if (
            row_overlap >= 2
            and row_overlap / len(row_token_set) >= 0.8
            and row_overlap / len(line_token_set) >= 0.8
        ):
            return True
    return False


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+(?:@\w+)?", text.lower())


def _normalize_line(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
