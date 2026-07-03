from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ragent_forge.app.models import Document, DocumentChunk
from ragent_forge.core.ingestion.document_blocks import DocumentBlock


class BlockChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        document: Document,
        blocks: Sequence[DocumentBlock],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        pending: list[DocumentBlock] = []

        for block in blocks:
            if not block.text:
                continue
            if block.block_type == "table":
                self._flush(document, pending, chunks)
                pending.clear()
                self._append_chunk(document, [block], chunks)
                continue

            pending_text = _joined_text([*pending, block])
            if pending and len(pending_text) > self.chunk_size:
                self._flush(document, pending, chunks)
                pending.clear()
            pending.append(block)

        self._flush(document, pending, chunks)
        return chunks

    def _flush(
        self,
        document: Document,
        blocks: list[DocumentBlock],
        chunks: list[DocumentChunk],
    ) -> None:
        if blocks:
            self._append_chunk(document, blocks, chunks)

    def _append_chunk(
        self,
        document: Document,
        blocks: Sequence[DocumentBlock],
        chunks: list[DocumentChunk],
    ) -> None:
        source_path = str(document.metadata.get("source_path", document.id))
        chunk_index = len(chunks)
        text = _joined_text(blocks)
        metadata = _chunk_metadata(source_path, text, self.chunk_size, blocks)
        chunks.append(
            DocumentChunk(
                id=f"{source_path}::chunk-{chunk_index:04d}",
                document_id=document.id,
                text=text,
                index=chunk_index,
                metadata=metadata,
            )
        )


def _joined_text(blocks: Sequence[DocumentBlock]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text)


def _chunk_metadata(
    source_path: str,
    text: str,
    chunk_size: int,
    blocks: Sequence[DocumentBlock],
) -> dict[str, Any]:
    pages = [block.page_number for block in blocks if block.page_number is not None]
    block_types = _unique([block.block_type for block in blocks])
    table_indices = _unique(
        [
            table_index
            for block in blocks
            for table_index in [_optional_int(block.metadata.get("table_index"))]
            if table_index is not None
        ]
    )
    warnings = [
        warning
        for block in blocks
        for warning in _warning_dicts(block.metadata.get("warnings"))
    ]
    metadata: dict[str, Any] = {
        "source_path": source_path,
        "media_type": "application/pdf",
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "block_types": block_types,
        "extraction_method": "pdf_structured",
        "warnings": warnings,
    }
    if len(block_types) == 1:
        metadata["block_type"] = block_types[0]
    if table_indices:
        metadata["table_indices"] = table_indices
    if len(text) > chunk_size:
        metadata["exceeds_chunk_size"] = True
    return metadata


def _unique(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _warning_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
