from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from ragent_forge.core.ingestion.document_blocks import DocumentBlock
from ragent_forge.core.ingestion.structured_loader import (
    PDF_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    load_structured_document,
)
from ragent_forge.core.ingestion.structured_result import StructuredLoadResult

DEFAULT_INCLUDE_BLOCK_TYPES = frozenset(
    {"paragraph", "list", "table", "code", "blockquote"}
)
T = TypeVar("T")


@dataclass(frozen=True)
class EvidenceSpan:
    id: str
    source_path: str
    document_id: str
    start_char: int | None
    end_char: int | None
    text: str
    media_type: str
    section_title: str | None
    heading_path: tuple[str, ...]
    block_types: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceSpanService:
    def __init__(
        self,
        min_chars: int = 250,
        max_chars: int = 1200,
        include_block_types: set[str] | None = None,
        include_pdf: bool = False,
    ) -> None:
        if min_chars < 1:
            raise ValueError("min_chars must be greater than 0")
        if max_chars < 1:
            raise ValueError("max_chars must be greater than 0")
        if max_chars < min_chars:
            raise ValueError("max_chars must be greater than or equal to min_chars")

        self.min_chars = min_chars
        self.max_chars = max_chars
        self.include_block_types = frozenset(
            DEFAULT_INCLUDE_BLOCK_TYPES
            if include_block_types is None
            else include_block_types
        )
        self.include_pdf = include_pdf
        self.supported_extensions = frozenset(
            SUPPORTED_EXTENSIONS
            if include_pdf
            else SUPPORTED_EXTENSIONS - PDF_EXTENSIONS
        )

    def extract(
        self,
        source_path: str | Path,
        max_spans: int | None = None,
    ) -> list[EvidenceSpan]:
        if max_spans is not None and max_spans < 0:
            raise ValueError("max_spans must be greater than or equal to 0")

        evidence_path = Path(source_path).expanduser()
        if not evidence_path.exists():
            raise FileNotFoundError(f"Evidence source path not found: {evidence_path}")

        if max_spans == 0:
            return []

        supported_files = [
            file_path
            for file_path in self._candidate_files(evidence_path)
            if file_path.suffix.lower() in self.supported_extensions
        ]
        if not supported_files:
            raise ValueError(
                f"No supported {self._supported_file_label()} files found under: "
                f"{evidence_path}"
            )

        spans: list[EvidenceSpan] = []
        for file_path in supported_files:
            structured_result = load_structured_document(file_path)
            for span in self._extract_document_spans(structured_result):
                spans.append(span)
                if max_spans is not None and len(spans) >= max_spans:
                    return spans

        return spans

    def _candidate_files(self, path: Path) -> list[Path]:
        if path.is_file():
            if _is_workspace_chunks_path(path):
                return []
            return [path]

        return sorted(
            (
                file_path
                for file_path in path.rglob("*")
                if file_path.is_file()
                and not _is_workspace_chunks_file(path, file_path)
            ),
            key=lambda file_path: (
                len(file_path.relative_to(path).parts),
                str(file_path.relative_to(path)).lower(),
            ),
        )

    def _extract_document_spans(
        self,
        structured_result: StructuredLoadResult,
    ) -> list[EvidenceSpan]:
        spans: list[EvidenceSpan] = []
        span_index = 0
        if _result_media_type(structured_result) == "application/pdf":
            block_groups = self._pdf_block_groups(structured_result.blocks)
        else:
            block_groups = self._useful_block_groups(structured_result.blocks)

        for block_group in block_groups:
            for window in self._block_windows(block_group):
                span = self._span_from_blocks(
                    structured_result=structured_result,
                    blocks=window,
                    span_index=span_index,
                )
                if span is None:
                    continue
                spans.append(span)
                span_index += 1

        return spans

    def _pdf_block_groups(
        self,
        blocks: Sequence[DocumentBlock],
    ) -> list[list[DocumentBlock]]:
        groups: list[list[DocumentBlock]] = []
        pending: list[DocumentBlock] = []
        pending_page: int | None = None

        for block in blocks:
            if not self._is_useful_block(block):
                if pending:
                    groups.append(pending)
                    pending = []
                    pending_page = None
                continue

            if block.block_type == "table":
                if pending:
                    groups.append(pending)
                    pending = []
                    pending_page = None
                groups.append([block])
                continue

            block_page = _block_page_number(block)
            if pending and block_page != pending_page:
                groups.append(pending)
                pending = []

            pending.append(block)
            pending_page = block_page

        if pending:
            groups.append(pending)

        return groups

    def _useful_block_groups(
        self,
        blocks: Sequence[DocumentBlock],
    ) -> list[list[DocumentBlock]]:
        groups: list[list[DocumentBlock]] = []
        current_group: list[DocumentBlock] = []
        current_key: tuple[str, ...] | None = None

        for block in blocks:
            if not self._is_useful_block(block):
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_key = None
                continue

            group_key = _group_key(block)
            if current_group and group_key != current_key:
                groups.append(current_group)
                current_group = []

            current_group.append(block)
            current_key = group_key

        if current_group:
            groups.append(current_group)

        return groups

    def _block_windows(
        self,
        blocks: Sequence[DocumentBlock],
    ) -> list[list[DocumentBlock]]:
        windows: list[list[DocumentBlock]] = []
        pending: list[DocumentBlock] = []

        for block in blocks:
            if not pending:
                pending.append(block)
                continue

            candidate = [*pending, block]
            if len(_joined_block_text(candidate)) <= self.max_chars:
                pending.append(block)
                continue

            windows.append(pending)
            pending = [block]

        if pending:
            windows.append(pending)

        return windows

    def _span_from_blocks(
        self,
        *,
        structured_result: StructuredLoadResult,
        blocks: Sequence[DocumentBlock],
        span_index: int,
    ) -> EvidenceSpan | None:
        if not blocks:
            return None

        media_type = _result_media_type(structured_result)
        if media_type is None:
            media_type = blocks[0].media_type

        text = _joined_block_text(blocks)
        if len(text) > self.max_chars:
            text, raw_trimmed = _trim_text(text, self.max_chars)
        else:
            raw_trimmed = False

        if len(text) < self.min_chars:
            return None

        source_path = _metadata_string(
            structured_result.document.metadata.get("source_path")
        )
        if source_path is None:
            source_path = blocks[0].source_path

        document_id = structured_result.document.id
        start_char = _optional_int(blocks[0].metadata.get("start_char"))
        end_char = _optional_int(blocks[-1].metadata.get("end_char"))
        if raw_trimmed and media_type == "application/pdf":
            start_char = None
            end_char = None
        elif raw_trimmed and start_char is not None:
            end_char = start_char + len(text)

        heading_path = _metadata_string_tuple(blocks[0].metadata.get("heading_path"))
        section_title = _metadata_string(blocks[0].metadata.get("section_title"))
        block_types = _unique_block_types(blocks)
        block_indices = [block.block_index for block in blocks]
        page_numbers = _page_numbers(blocks)
        page_start = min(page_numbers) if page_numbers else None
        page_end = max(page_numbers) if page_numbers else None
        offsets_available = start_char is not None and end_char is not None

        metadata: dict[str, Any] = {
            "source_path": source_path,
            "document_id": document_id,
            "media_type": media_type,
            "section_title": section_title,
            "heading_path": list(heading_path),
            "block_types": list(block_types),
            "block_indices": block_indices,
            "block_count": len(blocks),
            "start_block_index": block_indices[0],
            "end_block_index": block_indices[-1],
            "offsets_available": offsets_available,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        if media_type == "application/pdf":
            _copy_pdf_metadata(
                metadata,
                blocks,
                page_start=page_start,
                page_end=page_end,
                page_numbers=page_numbers,
            )
        if raw_trimmed:
            metadata["raw_char_trimmed"] = True

        return EvidenceSpan(
            id=f"{document_id}::span-{span_index:04d}",
            source_path=source_path,
            document_id=document_id,
            start_char=start_char,
            end_char=end_char,
            text=text,
            media_type=media_type,
            section_title=section_title,
            heading_path=heading_path,
            block_types=block_types,
            page_start=page_start,
            page_end=page_end,
            metadata=metadata,
        )

    def _is_useful_block(self, block: DocumentBlock) -> bool:
        return block.block_type in self.include_block_types and bool(block.text.strip())

    def _supported_file_label(self) -> str:
        if self.include_pdf:
            return "Markdown/TXT/PDF"
        return "Markdown/TXT"


def _copy_pdf_metadata(
    metadata: dict[str, Any],
    blocks: Sequence[DocumentBlock],
    *,
    page_start: int | None,
    page_end: int | None,
    page_numbers: list[int],
) -> None:
    metadata["page_start"] = page_start
    metadata["page_end"] = page_end
    metadata["page_numbers"] = page_numbers

    for key in ("extraction_method", "reading_order_strategy"):
        value = _coalesced_metadata_string(blocks, key)
        if value:
            metadata[key] = value

    reading_order_warning = _joined_unique_metadata_strings(
        blocks,
        "reading_order_warning",
    )
    if reading_order_warning:
        metadata["reading_order_warning"] = reading_order_warning

    warnings = [
        warning
        for block in blocks
        for warning in _warning_dicts(block.metadata.get("warnings"))
    ]
    if warnings:
        metadata["warnings"] = warnings

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

    for key in (
        "table_caption",
        "table_context",
        "table_context_strategy",
        "serialization",
        "table_text_dedup_strategy",
    ):
        value = _first_metadata_string(blocks, key)
        if value:
            metadata[key] = value

    for key in ("row_count", "column_count"):
        value = _first_metadata_int(blocks, key)
        if value is not None:
            metadata[key] = value

    possible_formula_lines = _unique(
        [
            line
            for block in blocks
            for line in _metadata_string_list(
                block.metadata.get("possible_formula_lines")
            )
        ]
    )
    if any(block.metadata.get("possible_formula") is True for block in blocks):
        metadata["possible_formula"] = True
    if possible_formula_lines:
        metadata["possible_formula_lines"] = possible_formula_lines[:10]

    table_dedup_removed_lines = _sum_metadata_int(
        blocks,
        "table_text_dedup_removed_lines",
    )
    if (
        any(block.metadata.get("table_text_dedup_applied") is True for block in blocks)
        or table_dedup_removed_lines
    ):
        metadata["table_text_dedup_applied"] = True
        metadata["table_text_dedup_removed_lines"] = table_dedup_removed_lines

    header_footer_removed_lines = _sum_metadata_int(
        blocks,
        "header_footer_removed_lines",
    )
    if (
        any(
            block.metadata.get("header_footer_filter_applied") is True
            for block in blocks
        )
        or header_footer_removed_lines
    ):
        metadata["header_footer_filter_applied"] = True
        metadata["header_footer_removed_lines"] = header_footer_removed_lines
        header_footer_candidates = _unique(
            [
                candidate
                for block in blocks
                for candidate in _metadata_string_list(
                    block.metadata.get("header_footer_candidates")
                )
            ]
        )
        if header_footer_candidates:
            metadata["header_footer_candidates"] = header_footer_candidates[:10]


def _joined_block_text(blocks: Sequence[DocumentBlock]) -> str:
    return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())


def _trim_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False

    cutoff = text.rfind(" ", 0, max_chars + 1)
    if cutoff < max_chars // 2:
        cutoff = max_chars
    return text[:cutoff].rstrip(), True


def _group_key(block: DocumentBlock) -> tuple[str, ...]:
    if block.media_type == "text/markdown":
        heading_path = _metadata_string_tuple(block.metadata.get("heading_path"))
        if heading_path:
            return heading_path
    return ()


def _result_media_type(result: StructuredLoadResult) -> str | None:
    media_type = _metadata_string(result.document.metadata.get("media_type"))
    if media_type:
        return media_type
    if result.blocks:
        return result.blocks[0].media_type
    return None


def _page_numbers(blocks: Sequence[DocumentBlock]) -> list[int]:
    return _unique(
        [
            page_number
            for block in blocks
            for page_number in [_block_page_number(block)]
            if page_number is not None
        ]
    )


def _block_page_number(block: DocumentBlock) -> int | None:
    if block.page_number is not None:
        return block.page_number
    return _optional_int(block.metadata.get("page_number"))


def _unique_block_types(blocks: Sequence[DocumentBlock]) -> tuple[str, ...]:
    block_types: list[str] = []
    for block in blocks:
        if block.block_type not in block_types:
            block_types.append(block.block_type)
    return tuple(block_types)


def _coalesced_metadata_string(
    blocks: Sequence[DocumentBlock],
    key: str,
) -> str | None:
    values = _unique(
        [
            value
            for block in blocks
            for value in [_metadata_string(block.metadata.get(key))]
            if value is not None
        ]
    )
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return "mixed"
    return None


def _joined_unique_metadata_strings(
    blocks: Sequence[DocumentBlock],
    key: str,
) -> str | None:
    values = _unique(
        [
            value
            for block in blocks
            for value in [_metadata_string(block.metadata.get(key))]
            if value is not None
        ]
    )
    if values:
        return "; ".join(values)
    return None


def _first_metadata_string(
    blocks: Sequence[DocumentBlock],
    key: str,
) -> str | None:
    for block in blocks:
        value = _metadata_string(block.metadata.get(key))
        if value:
            return value
    return None


def _first_metadata_int(
    blocks: Sequence[DocumentBlock],
    key: str,
) -> int | None:
    for block in blocks:
        value = _optional_int(block.metadata.get(key))
        if value is not None:
            return value
    return None


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


def _metadata_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _metadata_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str) and item)
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _metadata_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _warning_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    warnings: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            warnings.append(dict(item))
    return warnings


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _unique(values: Sequence[T]) -> list[T]:
    unique_values: list[T] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _is_workspace_chunks_file(root: Path, file_path: Path) -> bool:
    try:
        relative_parts = file_path.relative_to(root).parts
    except ValueError:
        return _is_workspace_chunks_path(file_path)
    return _parts_include_workspace_chunks(relative_parts)


def _is_workspace_chunks_path(path: Path) -> bool:
    return _parts_include_workspace_chunks(path.parts)


def _parts_include_workspace_chunks(parts: Sequence[str]) -> bool:
    lowered_parts = [part.lower() for part in parts]
    return any(
        part == ".ragent"
        and index + 1 < len(lowered_parts)
        and lowered_parts[index + 1] == "chunks"
        for index, part in enumerate(lowered_parts)
    )
