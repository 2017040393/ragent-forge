from __future__ import annotations

from pathlib import Path

from ragent_forge.core.ingestion.markdown_loader import (
    MARKDOWN_EXTENSIONS,
    TEXT_EXTENSIONS,
    load_markdown_document,
    load_text_document,
)
from ragent_forge.core.ingestion.pdf_loader import load_pdf_document
from ragent_forge.core.ingestion.structured_result import StructuredLoadResult

PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = MARKDOWN_EXTENSIONS | TEXT_EXTENSIONS | PDF_EXTENSIONS


def load_structured_document(path: str | Path) -> StructuredLoadResult:
    source_path = Path(path).expanduser()
    extension = source_path.suffix.lower()

    if extension in MARKDOWN_EXTENSIONS:
        return load_markdown_document(source_path)
    if extension in TEXT_EXTENSIONS:
        return load_text_document(source_path)
    if extension in PDF_EXTENSIONS:
        pdf_result = load_pdf_document(source_path)
        return StructuredLoadResult(
            document=pdf_result.document,
            blocks=pdf_result.blocks,
            metadata=pdf_result.metadata,
            warnings=pdf_result.warnings,
        )

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(
        f"Unsupported file type '{extension}' for {source_path}. "
        f"Supported file types: {supported}."
    )
