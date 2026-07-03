from pathlib import Path
from typing import Any

from ragent_forge.app.models import IngestResult
from ragent_forge.core.chunking.block_chunker import BlockChunker
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.core.ingestion.markdown_loader import (
    SUPPORTED_EXTENSIONS as TEXT_EXTENSIONS,
)
from ragent_forge.core.ingestion.markdown_loader import (
    load_document,
)
from ragent_forge.core.ingestion.pdf_loader import PdfLoadResult, load_pdf_document

PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


class IngestService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
        self.chunker = SimpleChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.block_chunker = BlockChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest(self, path: str | Path) -> IngestResult:
        ingest_path = Path(path).expanduser()

        if not ingest_path.exists():
            raise FileNotFoundError(f"Ingest path not found: {ingest_path}")

        candidate_files = self._candidate_files(ingest_path)
        supported_files = [
            file_path
            for file_path in candidate_files
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        skipped_files = [
            str(file_path.resolve())
            for file_path in candidate_files
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ]

        if not supported_files:
            raise ValueError(
                f"No supported Markdown/TXT/PDF files found under: {ingest_path}"
            )

        documents = []
        chunks = []
        pdf_results: list[PdfLoadResult] = []
        for file_path in supported_files:
            if file_path.suffix.lower() in PDF_EXTENSIONS:
                pdf_result = load_pdf_document(file_path)
                pdf_results.append(pdf_result)
                documents.append(pdf_result.document)
                chunks.extend(
                    self.block_chunker.chunk(
                        pdf_result.document,
                        pdf_result.blocks,
                    )
                )
            else:
                document = load_document(file_path)
                documents.append(document)
                chunks.extend(self.chunker.chunk(document))

        if not chunks:
            raise ValueError(
                f"No chunks produced from supported files under: {ingest_path}"
            )

        metadata: dict[str, Any] = {
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
        }
        if pdf_results:
            metadata["pdf"] = _pdf_summary(pdf_results)

        return IngestResult(
            source_path=str(ingest_path.resolve()),
            documents=documents,
            chunks=chunks,
            skipped_files=skipped_files,
            metadata=metadata,
        )

    def _candidate_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]

        return sorted(
            (file_path for file_path in path.rglob("*") if file_path.is_file()),
            key=lambda file_path: (
                len(file_path.relative_to(path).parts),
                str(file_path.relative_to(path)).lower(),
            ),
        )


def _pdf_summary(pdf_results: list[PdfLoadResult]) -> dict[str, Any]:
    warnings = [
        warning.to_dict()
        for result in pdf_results
        for warning in result.warnings
    ]
    return {
        "pdf_files_seen": len(pdf_results),
        "pdf_files_ingested": len(pdf_results),
        "pdf_pages_seen": sum(
            int(result.metadata.get("page_count", 0) or 0)
            for result in pdf_results
        ),
        "pdf_pages_with_text": sum(
            int(result.metadata.get("pages_with_text", 0) or 0)
            for result in pdf_results
        ),
        "pdf_tables_extracted": sum(
            int(result.metadata.get("tables_extracted", 0) or 0)
            for result in pdf_results
        ),
        "pdf_empty_pages": sum(
            int(result.metadata.get("empty_pages", 0) or 0)
            for result in pdf_results
        ),
        "pdf_reading_order_strategy": _pdf_reading_order_strategy(pdf_results),
        "pdf_reading_order_fallback_pages": _pdf_metadata_sum(
            pdf_results,
            "reading_order_fallback_pages",
        ),
        "pdf_reading_order_warnings": [
            warning
            for result in pdf_results
            for warning in _metadata_list(
                result.metadata.get("reading_order_warnings")
            )
        ],
        "pdf_table_text_dedup_pages": _pdf_metadata_sum(
            pdf_results,
            "table_text_dedup_pages",
        ),
        "pdf_table_text_dedup_removed_lines": _pdf_metadata_sum(
            pdf_results,
            "table_text_dedup_removed_lines",
        ),
        "pdf_possible_formula_blocks": _pdf_metadata_sum(
            pdf_results,
            "possible_formula_blocks",
        ),
        "pdf_possible_formula_lines": _pdf_metadata_sum(
            pdf_results,
            "possible_formula_lines",
        ),
        "pdf_suspected_headers_filtered": _pdf_metadata_sum(
            pdf_results,
            "suspected_headers_filtered",
        ),
        "pdf_suspected_footers_filtered": _pdf_metadata_sum(
            pdf_results,
            "suspected_footers_filtered",
        ),
        "pdf_warnings": warnings,
    }


def _pdf_reading_order_strategy(pdf_results: list[PdfLoadResult]) -> str:
    strategies = [
        str(result.metadata.get("reading_order_strategy"))
        for result in pdf_results
        if result.metadata.get("reading_order_strategy")
    ]
    if "coordinate_blocks" in strategies:
        return "coordinate_blocks"
    if "pdfplumber_words" in strategies:
        return "pdfplumber_words"
    if strategies:
        return strategies[0]
    return "pdfplumber_words"


def _pdf_metadata_sum(pdf_results: list[PdfLoadResult], key: str) -> int:
    return sum(int(result.metadata.get(key, 0) or 0) for result in pdf_results)


def _metadata_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []
