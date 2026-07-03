from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber

from ragent_forge.app.models import Document
from ragent_forge.core.ingestion.document_blocks import (
    DocumentBlock,
    PdfExtractionWarning,
)
from ragent_forge.core.ingestion.table_serialization import serialize_table


@dataclass(frozen=True)
class PdfLoadResult:
    document: Document
    blocks: tuple[DocumentBlock, ...]
    warnings: tuple[PdfExtractionWarning, ...]
    metadata: dict[str, Any]


def load_pdf_document(path: str | Path) -> PdfLoadResult:
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF document not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"PDF document path is not a file: {source_path}")
    if source_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported PDF file type '{source_path.suffix.lower()}'")

    resolved_path = source_path.resolve()
    blocks: list[DocumentBlock] = []
    warnings: list[PdfExtractionWarning] = []
    page_count = 0
    pages_with_text = 0
    tables_extracted = 0
    empty_pages = 0

    try:
        with pdfplumber.open(resolved_path) as pdf:
            page_count = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages, start=1):
                page_had_content = False
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages_with_text += 1
                    page_had_content = True
                    blocks.append(
                        DocumentBlock(
                            source_path=str(resolved_path),
                            media_type="application/pdf",
                            page_number=page_index,
                            block_index=len(blocks),
                            block_type="paragraph",
                            text=page_text,
                            metadata={
                                "page_number": page_index,
                                "extraction_method": "pdfplumber",
                                "media_type": "application/pdf",
                            },
                        )
                    )

                page_tables, table_warnings = _extract_page_tables(
                    page=page,
                    source_path=str(resolved_path),
                    page_number=page_index,
                    starting_block_index=len(blocks),
                )
                if page_tables:
                    page_had_content = True
                    tables_extracted += len(page_tables)
                    blocks.extend(page_tables)
                warnings.extend(table_warnings)

                if not page_had_content:
                    empty_pages += 1
                    message = "No extractable text or usable table found on page."
                    warnings.append(
                        PdfExtractionWarning(
                            source_path=str(resolved_path),
                            page=page_index,
                            kind="empty_page",
                            message=message,
                        )
                    )
    except Exception as exc:
        raise ValueError(f"Failed to read PDF {resolved_path}: {exc}") from exc

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
    starting_block_index: int,
) -> tuple[list[DocumentBlock], list[PdfExtractionWarning]]:
    blocks: list[DocumentBlock] = []
    warnings: list[PdfExtractionWarning] = []
    tables = page.extract_tables() or []
    for table_index, table in enumerate(tables, start=1):
        serialized = serialize_table(table)
        table_warnings = [
            PdfExtractionWarning(
                source_path=source_path,
                page=page_number,
                kind=warning_kind,
                message=_warning_message(warning_kind),
            )
            for warning_kind in serialized.warning_kinds
        ]
        warnings.extend(table_warnings)
        if not serialized.text:
            continue
        blocks.append(
            DocumentBlock(
                source_path=source_path,
                media_type="application/pdf",
                page_number=page_number,
                block_index=starting_block_index + len(blocks),
                block_type="table",
                text=serialized.text,
                metadata={
                    "page_number": page_number,
                    "table_index": table_index,
                    "row_count": serialized.row_count,
                    "column_count": serialized.column_count,
                    "serialization": serialized.serialization,
                    "extraction_method": "pdfplumber",
                    "media_type": "application/pdf",
                    "warnings": [warning.to_dict() for warning in table_warnings],
                },
            )
        )
    return blocks, warnings


def _warning_message(kind: str) -> str:
    if kind == "table_empty":
        return "Extracted table had no usable cells."
    if kind == "table_malformed":
        return "Extracted table had inconsistent row widths."
    return "PDF extraction warning."
