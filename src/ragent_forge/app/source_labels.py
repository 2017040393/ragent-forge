from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def format_source_label(
    source_path: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    label = _basename(source_path)
    if not _is_pdf_metadata(metadata):
        return label

    page_start = _optional_int(metadata.get("page_start"))
    page_end = _optional_int(metadata.get("page_end"))
    if page_start is None:
        return label
    if page_end is None:
        page_end = page_start

    if page_start == page_end:
        label = f"{label} p.{page_start}"
    else:
        label = f"{label} pp.{page_start}-{page_end}"

    table_index = _single_int(metadata.get("table_indices"))
    if table_index is not None:
        label = f"{label} table {table_index}"
    return label


def format_pdf_source_metadata(metadata: Mapping[str, Any] | None) -> list[str]:
    if not _is_pdf_metadata(metadata):
        return []

    lines = ["type: pdf"]
    page_range = _page_range_text(metadata)
    if page_range:
        lines.append(f"page range: {page_range}")

    block_type = _block_type_text(metadata)
    if block_type:
        lines.append(f"block type: {block_type}")

    table_text = _table_text(metadata)
    if table_text:
        lines.append(f"table: {table_text}")

    table_caption = metadata.get("table_caption")
    if isinstance(table_caption, str) and table_caption:
        lines.append(f"table caption: {table_caption}")

    reading_order = metadata.get("reading_order_strategy")
    if isinstance(reading_order, str) and reading_order:
        lines.append(f"reading order: {reading_order}")

    if metadata.get("table_text_dedup_applied") is True:
        lines.append("dedup: applied")

    if metadata.get("possible_formula") is True:
        lines.append("possible formula: yes")

    if metadata.get("header_footer_filter_applied") is True:
        lines.append("header/footer: filtered")

    warning_text = _warning_text(metadata)
    if warning_text:
        lines.append(f"warnings: {warning_text}")
    return lines


def format_source_range(
    start_char: int | None,
    end_char: int | None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    if isinstance(start_char, int) and isinstance(end_char, int):
        return f"{start_char}-{end_char}"
    if not _is_pdf_metadata(metadata):
        return ""

    page_start = _optional_int(metadata.get("page_start"))
    page_end = _optional_int(metadata.get("page_end"))
    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return f"p.{page_start}"
    return f"pp.{page_start}-{page_end}"


def _basename(path: str) -> str:
    normalized = path.replace("\\", "/")
    name = normalized.rstrip("/").rsplit("/", 1)[-1]
    return name or path


def _is_pdf_metadata(metadata: Mapping[str, Any] | None) -> bool:
    return bool(metadata and metadata.get("media_type") == "application/pdf")


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _single_int(value: Any) -> int | None:
    if not isinstance(value, list) or len(value) != 1:
        return None
    return _optional_int(value[0])


def _page_range_text(metadata: Mapping[str, Any]) -> str:
    page_start = _optional_int(metadata.get("page_start"))
    page_end = _optional_int(metadata.get("page_end"))
    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _block_type_text(metadata: Mapping[str, Any]) -> str:
    block_type = metadata.get("block_type")
    if isinstance(block_type, str):
        return block_type
    block_types = metadata.get("block_types")
    if isinstance(block_types, list):
        return ", ".join(str(item) for item in block_types if item)
    return ""


def _table_text(metadata: Mapping[str, Any]) -> str:
    table_indices = metadata.get("table_indices")
    if isinstance(table_indices, list):
        values = [str(item) for item in table_indices if isinstance(item, int)]
        return ", ".join(values)
    return ""


def _warning_text(metadata: Mapping[str, Any]) -> str:
    warnings = metadata.get("warnings")
    if not isinstance(warnings, list):
        return ""
    kinds: list[str] = []
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        kind = warning.get("kind")
        if isinstance(kind, str) and kind not in kinds:
            kinds.append(kind)
    return ", ".join(kinds)
