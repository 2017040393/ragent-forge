from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pdfplumber

from ragent_forge.core.ingestion.document_blocks import (
    DocumentBlock,
    PdfExtractionWarning,
)
from ragent_forge.core.ingestion.pdf_formula_detection import detect_formula_lines
from ragent_forge.core.ingestion.pdf_reading_order import (
    PdfPageText,
    PdfTextLine,
    extract_ordered_page_text,
)
from ragent_forge.core.ingestion.pdf_text_filters import (
    TableDedupResult,
    filter_repeated_header_footer_lines,
    remove_table_text_duplicates,
)
from ragent_forge.core.ingestion.table_serialization import (
    TableSerializationResult,
    serialize_table,
)
from ragent_forge.core.models import Document

_TableWithBbox = tuple[
    Sequence[Sequence[Any] | None],
    tuple[float, float, float, float] | None,
]


@dataclass(frozen=True)
class PdfLoadResult:
    document: Document
    blocks: tuple[DocumentBlock, ...]
    warnings: tuple[PdfExtractionWarning, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ExtractedTable:
    table_index: int
    raw_table: Sequence[Sequence[Any] | None]
    serialized: TableSerializationResult
    bbox: tuple[float, float, float, float] | None
    warnings: tuple[PdfExtractionWarning, ...]


@dataclass(frozen=True)
class _PageRecord:
    page_number: int
    ordered_text: PdfPageText
    tables: tuple[_ExtractedTable, ...]


_CAPTION_PATTERN = re.compile(
    r"^\s*(?:Table\s+\d+[:.]?|TABLE\s+[IVXLCDM]+[:.]?|\u8868\s*\d+[:.]?).+",
    re.IGNORECASE,
)


def load_pdf_document(path: str | Path) -> PdfLoadResult:
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF document not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"PDF document path is not a file: {source_path}")
    if source_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported PDF file type '{source_path.suffix.lower()}'")

    resolved_path = source_path.resolve()
    page_records: list[_PageRecord] = []
    warnings: list[PdfExtractionWarning] = []

    try:
        with pdfplumber.open(resolved_path) as pdf:
            page_count = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages, start=1):
                ordered_text = extract_ordered_page_text(page)
                if ordered_text.fallback_used:
                    warning = PdfExtractionWarning(
                        source_path=str(resolved_path),
                        page=page_index,
                        kind="reading_order_fallback",
                        message=(
                            ordered_text.warning
                            or "PDF page text used extract_text fallback."
                        ),
                    )
                    warnings.append(warning)

                page_tables, table_warnings = _extract_page_tables(
                    page=page,
                    source_path=str(resolved_path),
                    page_number=page_index,
                )
                warnings.extend(table_warnings)
                page_records.append(
                    _PageRecord(
                        page_number=page_index,
                        ordered_text=ordered_text,
                        tables=tuple(page_tables),
                    )
                )
    except Exception as exc:
        raise ValueError(f"Failed to read PDF {resolved_path}: {exc}") from exc

    blocks: list[DocumentBlock] = []
    pages_with_text = 0
    tables_extracted = 0
    empty_pages = 0
    table_text_dedup_pages = 0
    table_text_dedup_removed_lines = 0
    possible_formula_blocks = 0
    possible_formula_lines = 0

    header_footer_result = filter_repeated_header_footer_lines(
        [record.ordered_text.lines for record in page_records]
    )

    for record, header_footer_page in zip(
        page_records,
        header_footer_result.pages,
        strict=True,
    ):
        page_had_content = False
        dedup = remove_table_text_duplicates(
            header_footer_page.text,
            [table.raw_table for table in record.tables],
        )
        page_text = dedup.text.strip()
        formula_lines = detect_formula_lines(page_text)

        if page_text:
            pages_with_text += 1
            page_had_content = True
            if dedup.removed_lines:
                table_text_dedup_pages += 1
                table_text_dedup_removed_lines += dedup.removed_lines
            if formula_lines:
                possible_formula_blocks += 1
                possible_formula_lines += len(formula_lines)

            blocks.append(
                DocumentBlock(
                    source_path=str(resolved_path),
                    media_type="application/pdf",
                    page_number=record.page_number,
                    block_index=len(blocks),
                    block_type="paragraph",
                    text=page_text,
                    metadata=_paragraph_metadata(
                        page_number=record.page_number,
                        ordered_text=record.ordered_text,
                        dedup=dedup,
                        formula_lines=formula_lines,
                        header_removed_lines=(
                            header_footer_page.header_removed_lines
                        ),
                        footer_removed_lines=(
                            header_footer_page.footer_removed_lines
                        ),
                        header_footer_candidates=header_footer_page.candidates,
                    ),
                )
            )

        for table in record.tables:
            if not table.serialized.text:
                continue
            page_had_content = True
            tables_extracted += 1
            blocks.append(
                _table_block(
                    source_path=str(resolved_path),
                    page_number=record.page_number,
                    block_index=len(blocks),
                    table=table,
                    page_lines=record.ordered_text.lines,
                )
            )

        if not page_had_content:
            empty_pages += 1
            warnings.append(
                PdfExtractionWarning(
                    source_path=str(resolved_path),
                    page=record.page_number,
                    kind="empty_page",
                    message="No extractable text or usable table found on page.",
                )
            )

    text = "\n\n".join(block.text for block in blocks)
    metadata: dict[str, Any] = {
        "source_path": str(resolved_path),
        "file_name": resolved_path.name,
        "extension": ".pdf",
        "media_type": "application/pdf",
        "page_count": page_count,
        "pages_with_text": pages_with_text,
        "tables_extracted": tables_extracted,
        "empty_pages": empty_pages,
        "character_count": len(text),
        "reading_order_strategy": _aggregate_reading_order_strategy(page_records),
        "reading_order_fallback_pages": sum(
            1 for record in page_records if record.ordered_text.fallback_used
        ),
        "reading_order_warnings": [
            warning.to_dict()
            for warning in warnings
            if warning.kind.startswith("reading_order")
        ],
        "table_text_dedup_pages": table_text_dedup_pages,
        "table_text_dedup_removed_lines": table_text_dedup_removed_lines,
        "possible_formula_blocks": possible_formula_blocks,
        "possible_formula_lines": possible_formula_lines,
        "suspected_headers_filtered": (
            header_footer_result.suspected_headers_filtered
        ),
        "suspected_footers_filtered": (
            header_footer_result.suspected_footers_filtered
        ),
    }
    document = Document(id=str(resolved_path), text=text, metadata=metadata)
    return PdfLoadResult(
        document=document,
        blocks=tuple(blocks),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _extract_page_tables(
    *,
    page: Any,
    source_path: str,
    page_number: int,
) -> tuple[list[_ExtractedTable], list[PdfExtractionWarning]]:
    tables: list[_ExtractedTable] = []
    warnings: list[PdfExtractionWarning] = []
    for table_index, (raw_table, bbox) in enumerate(_find_page_tables(page), start=1):
        serialized = serialize_table(raw_table)
        table_warnings = tuple(
            PdfExtractionWarning(
                source_path=source_path,
                page=page_number,
                kind=warning_kind,
                message=_warning_message(warning_kind),
            )
            for warning_kind in serialized.warning_kinds
        )
        warnings.extend(table_warnings)
        if not serialized.text:
            continue
        tables.append(
            _ExtractedTable(
                table_index=table_index,
                raw_table=raw_table,
                serialized=serialized,
                bbox=bbox,
                warnings=table_warnings,
            )
        )
    return tables, warnings


def _find_page_tables(
    page: Any,
) -> list[_TableWithBbox]:
    try:
        table_objects = page.find_tables() or []
    except Exception:
        table_objects = []
    positioned_tables: list[_TableWithBbox] = []
    for table in table_objects:
        raw_table = table.extract() or []
        if not raw_table:
            continue
        bbox = cast(
            tuple[float, float, float, float] | None,
            tuple(table.bbox) if table.bbox else None,
        )
        positioned_tables.append((raw_table, bbox))
    if positioned_tables:
        return positioned_tables

    extracted_tables: list[_TableWithBbox] = [
        (table, None) for table in page.extract_tables() or [] if table
    ]
    return extracted_tables


def _paragraph_metadata(
    *,
    page_number: int,
    ordered_text: PdfPageText,
    dedup: TableDedupResult,
    formula_lines: list[str],
    header_removed_lines: int,
    footer_removed_lines: int,
    header_footer_candidates: Sequence[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "page_number": page_number,
        "extraction_method": "pdfplumber",
        "media_type": "application/pdf",
        "reading_order_strategy": ordered_text.strategy,
    }
    if ordered_text.warning:
        metadata["reading_order_warning"] = ordered_text.warning
    if dedup.removed_lines:
        metadata["table_text_dedup_applied"] = True
        metadata["table_text_dedup_strategy"] = dedup.strategy
        metadata["table_text_dedup_removed_lines"] = dedup.removed_lines
    if formula_lines:
        metadata["possible_formula"] = True
        metadata["possible_formula_lines"] = formula_lines[:10]
    removed_header_footer = header_removed_lines + footer_removed_lines
    if removed_header_footer:
        metadata["header_footer_filter_applied"] = True
        metadata["header_footer_removed_lines"] = removed_header_footer
        metadata["header_footer_candidates"] = list(header_footer_candidates)[:10]
    return metadata


def _table_block(
    *,
    source_path: str,
    page_number: int,
    block_index: int,
    table: _ExtractedTable,
    page_lines: Sequence[PdfTextLine],
) -> DocumentBlock:
    table_caption = _find_table_caption(page_lines, table.bbox)
    text = table.serialized.text
    metadata: dict[str, Any] = {
        "page_number": page_number,
        "table_index": table.table_index,
        "row_count": table.serialized.row_count,
        "column_count": table.serialized.column_count,
        "serialization": table.serialized.serialization,
        "extraction_method": "pdfplumber",
        "media_type": "application/pdf",
        "warnings": [warning.to_dict() for warning in table.warnings],
    }
    if table_caption:
        text = f"{table_caption}\n\n{text}"
        metadata["table_caption"] = table_caption
        metadata["table_context_strategy"] = "same_page_caption_before_table"

    return DocumentBlock(
        source_path=source_path,
        media_type="application/pdf",
        page_number=page_number,
        block_index=block_index,
        block_type="table",
        text=text,
        metadata=metadata,
    )


def _find_table_caption(
    lines: Sequence[PdfTextLine],
    bbox: tuple[float, float, float, float] | None,
) -> str | None:
    caption_lines = [line for line in lines if _CAPTION_PATTERN.match(line.text)]
    if not caption_lines:
        return None
    if bbox is None:
        return caption_lines[0].text

    table_top = bbox[1]
    before_table = [line for line in caption_lines if line.top < table_top]
    if not before_table:
        return None
    return max(before_table, key=lambda line: line.top).text


def _aggregate_reading_order_strategy(page_records: Sequence[_PageRecord]) -> str:
    strategies = [
        record.ordered_text.strategy
        for record in page_records
        if record.ordered_text.text
    ]
    if "coordinate_blocks" in strategies:
        return "coordinate_blocks"
    if "pdfplumber_words" in strategies:
        return "pdfplumber_words"
    if strategies:
        return strategies[0]
    return "pdfplumber_words"


def _warning_message(kind: str) -> str:
    if kind == "table_empty":
        return "Extracted table had no usable cells."
    if kind == "table_malformed":
        return "Extracted table had inconsistent row widths."
    return "PDF extraction warning."
