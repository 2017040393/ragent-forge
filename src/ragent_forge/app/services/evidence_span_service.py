from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

        for block_group in self._useful_block_groups(structured_result.blocks):
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
        media_type = _metadata_string(
            structured_result.document.metadata.get("media_type")
        )
        if media_type is None:
            media_type = blocks[0].media_type

        start_char = _optional_int(blocks[0].metadata.get("start_char"))
        end_char = _optional_int(blocks[-1].metadata.get("end_char"))
        if raw_trimmed and start_char is not None:
            end_char = start_char + len(text)

        heading_path = _metadata_string_tuple(blocks[0].metadata.get("heading_path"))
        section_title = _metadata_string(blocks[0].metadata.get("section_title"))
        block_types = _unique_block_types(blocks)
        block_indices = [block.block_index for block in blocks]

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
        }
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
            metadata=metadata,
        )

    def _is_useful_block(self, block: DocumentBlock) -> bool:
        return block.block_type in self.include_block_types and bool(block.text.strip())

    def _supported_file_label(self) -> str:
        if self.include_pdf:
            return "Markdown/TXT/PDF"
        return "Markdown/TXT"


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


def _unique_block_types(blocks: Sequence[DocumentBlock]) -> tuple[str, ...]:
    block_types: list[str] = []
    for block in blocks:
        if block.block_type not in block_types:
            block_types.append(block.block_type)
    return tuple(block_types)


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


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


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
