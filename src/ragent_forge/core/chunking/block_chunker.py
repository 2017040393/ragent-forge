from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ragent_forge.core.ingestion.document_blocks import DocumentBlock
from ragent_forge.core.models import Document, DocumentChunk


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
            if len(block.text) > self.chunk_size:
                self._flush(document, pending, chunks)
                pending.clear()
                self._split_oversized_block(document, block, chunks)
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

    def _split_oversized_block(
        self,
        document: Document,
        block: DocumentBlock,
        chunks: list[DocumentChunk],
    ) -> None:
        step = self.chunk_size - self.chunk_overlap
        start = 0
        block_start_char = _optional_int(block.metadata.get("start_char"))
        while start < len(block.text):
            end = min(start + self.chunk_size, len(block.text))
            char_range = (
                (block_start_char + start, block_start_char + end)
                if block_start_char is not None
                else None
            )
            self._append_chunk(
                document,
                [block],
                chunks,
                text_override=block.text[start:end],
                char_range_override=char_range,
            )
            if end == len(block.text):
                break
            start += step

    def _append_chunk(
        self,
        document: Document,
        blocks: Sequence[DocumentBlock],
        chunks: list[DocumentChunk],
        text_override: str | None = None,
        char_range_override: tuple[int, int] | None = None,
    ) -> None:
        source_path = str(document.metadata.get("source_path", document.id))
        chunk_index = len(chunks)
        text = text_override if text_override is not None else _joined_text(blocks)
        metadata = _chunk_metadata(
            source_path,
            text,
            self.chunk_size,
            blocks,
            char_range_override=char_range_override,
        )
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
    *,
    char_range_override: tuple[int, int] | None = None,
) -> dict[str, Any]:
    media_type = _media_type(blocks)
    block_types = _unique([block.block_type for block in blocks])
    metadata: dict[str, Any] = {
        "source_path": source_path,
        "media_type": media_type,
        "block_types": block_types,
    }
    if len(block_types) == 1:
        metadata["block_type"] = block_types[0]
    _copy_character_range_metadata(
        metadata,
        blocks,
        char_range_override=char_range_override,
    )
    _copy_section_metadata(metadata, blocks)
    _copy_general_metadata(metadata, blocks)
    _copy_warning_metadata(metadata, blocks, include_empty=_is_pdf(media_type))

    if _is_pdf(media_type):
        _copy_pdf_metadata(metadata, blocks, chunk_text=text)

    if len(text) > chunk_size:
        metadata["exceeds_chunk_size"] = True
    return metadata


def _media_type(blocks: Sequence[DocumentBlock]) -> str:
    for block in blocks:
        if block.media_type:
            return block.media_type
    for block in blocks:
        media_type = _optional_str(block.metadata.get("media_type"))
        if media_type:
            return media_type
    return "application/octet-stream"


def _is_pdf(media_type: str) -> bool:
    return media_type == "application/pdf"


def _copy_character_range_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
    *,
    char_range_override: tuple[int, int] | None,
) -> None:
    if char_range_override is not None:
        metadata["start_char"] = char_range_override[0]
        metadata["end_char"] = char_range_override[1]
        return

    ranges = [
        (start_char, end_char)
        for block in blocks
        for start_char, end_char in [
            (
                _optional_int(block.metadata.get("start_char")),
                _optional_int(block.metadata.get("end_char")),
            )
        ]
        if start_char is not None and end_char is not None
    ]
    if ranges:
        metadata["start_char"] = min(start for start, _ in ranges)
        metadata["end_char"] = max(end for _, end in ranges)


def _copy_section_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
) -> None:
    section_title = _first_metadata_string(blocks, "section_title")
    if section_title:
        metadata["section_title"] = section_title

    heading_path = _first_metadata_string_list(blocks, "heading_path")
    if heading_path:
        metadata["heading_path"] = heading_path


def _copy_general_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
) -> None:
    for key in ("serialization", "code_language"):
        value = _first_metadata_string(blocks, key)
        if value:
            metadata[key] = value

    heading_levels = _unique(
        [
            heading_level
            for block in blocks
            for heading_level in [_optional_int(block.metadata.get("heading_level"))]
            if heading_level is not None
        ]
    )
    if len(heading_levels) == 1:
        metadata["heading_level"] = heading_levels[0]


def _copy_warning_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
    *,
    include_empty: bool,
) -> None:
    warnings = [
        warning
        for block in blocks
        for warning in _warning_dicts(block.metadata.get("warnings"))
    ]
    if warnings or include_empty:
        metadata["warnings"] = warnings


def _copy_pdf_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
    *,
    chunk_text: str,
) -> None:
    pages = [block.page_number for block in blocks if block.page_number is not None]
    if pages:
        metadata["page_start"] = min(pages)
        metadata["page_end"] = max(pages)
    metadata["extraction_method"] = "pdf_structured"

    table_indices = _unique(
        [
            table_index
            for block in blocks
            for table_index in [_optional_int(block.metadata.get("table_index"))]
            if table_index is not None
        ]
    )
    if table_indices:
        metadata["table_indices"] = table_indices

    _copy_reading_order_metadata(metadata, blocks)
    _copy_table_context_metadata(metadata, blocks)
    _copy_table_dedup_metadata(metadata, blocks)
    _copy_formula_metadata(metadata, blocks, chunk_text=chunk_text)
    _copy_header_footer_metadata(metadata, blocks)


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


def _copy_reading_order_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
) -> None:
    strategies = _unique(
        [
            strategy
            for block in blocks
            for strategy in [
                _optional_str(block.metadata.get("reading_order_strategy"))
            ]
            if strategy is not None
        ]
    )
    if len(strategies) == 1:
        metadata["reading_order_strategy"] = strategies[0]
    elif len(strategies) > 1:
        metadata["reading_order_strategy"] = "mixed"

    warnings = _unique(
        [
            warning
            for block in blocks
            for warning in [_optional_str(block.metadata.get("reading_order_warning"))]
            if warning is not None
        ]
    )
    if warnings:
        metadata["reading_order_warning"] = "; ".join(warnings)


def _copy_table_context_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
) -> None:
    for key in ("table_caption", "table_context", "table_context_strategy"):
        value = _first_metadata_string(blocks, key)
        if value:
            metadata[key] = value


def _copy_table_dedup_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
) -> None:
    removed_lines = _sum_metadata_int(blocks, "table_text_dedup_removed_lines")
    if removed_lines:
        metadata["table_text_dedup_applied"] = True
        metadata["table_text_dedup_removed_lines"] = removed_lines
        strategy = _first_metadata_string(blocks, "table_text_dedup_strategy")
        if strategy:
            metadata["table_text_dedup_strategy"] = strategy


def _copy_formula_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
    *,
    chunk_text: str,
) -> None:
    formula_lines = _unique(
        [
            line
            for block in blocks
            for line in _metadata_string_list(
                block.metadata.get("possible_formula_lines")
            )
        ]
    )
    if formula_lines:
        matching_formula_lines = [
            formula_line
            for formula_line in formula_lines
            if formula_line in chunk_text
        ]
        if matching_formula_lines:
            metadata["possible_formula"] = True
            metadata["possible_formula_lines"] = matching_formula_lines[:10]
        return

    has_formula = any(
        block.metadata.get("possible_formula") is True for block in blocks
    )
    if has_formula:
        metadata["possible_formula"] = True


def _copy_header_footer_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
) -> None:
    removed_lines = _sum_metadata_int(blocks, "header_footer_removed_lines")
    if removed_lines:
        metadata["header_footer_filter_applied"] = True
        metadata["header_footer_removed_lines"] = removed_lines
    candidates = _unique(
        [
            candidate
            for block in blocks
            for candidate in _metadata_string_list(
                block.metadata.get("header_footer_candidates")
            )
        ]
    )
    if candidates:
        metadata["header_footer_candidates"] = candidates[:10]


def _first_metadata_string(
    blocks: Sequence[DocumentBlock],
    key: str,
) -> str | None:
    for block in blocks:
        value = _optional_str(block.metadata.get(key))
        if value:
            return value
    return None


def _first_metadata_string_list(
    blocks: Sequence[DocumentBlock],
    key: str,
) -> list[str]:
    for block in blocks:
        values = _metadata_string_list(block.metadata.get(key))
        if values:
            return values
    return []


def _sum_metadata_int(
    blocks: Sequence[DocumentBlock],
    key: str,
) -> int:
    return sum(
        value
        for block in blocks
        for value in [_optional_int(block.metadata.get(key))]
        if value is not None
    )


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _metadata_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _warning_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
