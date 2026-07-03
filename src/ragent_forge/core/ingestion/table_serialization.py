from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TableSerializationResult:
    text: str
    serialization: str
    row_count: int
    column_count: int
    warning_kinds: tuple[str, ...] = ()


def serialize_table(table: Sequence[Sequence[Any] | None]) -> TableSerializationResult:
    rows = _normalize_rows(table)
    if not rows:
        return TableSerializationResult(
            text="",
            serialization="empty",
            row_count=0,
            column_count=0,
            warning_kinds=("table_empty",),
        )

    original_widths = [len(row) for row in rows]
    column_count = max(original_widths)
    warning_kinds = (
        ("table_malformed",)
        if any(width != column_count for width in original_widths)
        else ()
    )
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]
    if not any(header):
        header = [f"Column {index}" for index in range(1, column_count + 1)]

    markdown_rows = [
        _markdown_row(header),
        "|" + "|".join(["---"] * column_count) + "|",
        *(_markdown_row(row) for row in body),
    ]
    return TableSerializationResult(
        text="\n".join(markdown_rows),
        serialization="markdown_table",
        row_count=len(normalized),
        column_count=column_count,
        warning_kinds=warning_kinds,
    )


def _normalize_rows(table: Sequence[Sequence[Any] | None]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table:
        if row is None:
            continue
        normalized = [_normalize_cell(cell) for cell in row]
        if any(cell for cell in normalized):
            rows.append(normalized)
    return rows


def _normalize_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def _markdown_row(row: Sequence[str]) -> str:
    return "| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |"


def _escape_markdown_cell(cell: str) -> str:
    return cell.replace("|", "\\|")
