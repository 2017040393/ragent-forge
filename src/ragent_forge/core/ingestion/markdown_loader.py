from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Never

from ragent_forge.core.ingestion.document_blocks import BlockType, DocumentBlock
from ragent_forge.core.ingestion.structured_result import StructuredLoadResult
from ragent_forge.core.models import Document

MARKDOWN_EXTENSIONS = {".md"}
TEXT_EXTENSIONS = {".txt"}
SUPPORTED_EXTENSIONS = MARKDOWN_EXTENSIONS | TEXT_EXTENSIONS

MARKDOWN_MEDIA_TYPE = "text/markdown"
TEXT_MEDIA_TYPE = "text/plain"

_ATX_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)\s*(\S*)?.*$")


def load_document(path: str | Path) -> Document:
    result = _load_supported_document(path)
    return result.document


def load_markdown_document(path: str | Path) -> StructuredLoadResult:
    source_path = _validate_supported_file(path, MARKDOWN_EXTENSIONS)
    text = source_path.read_text(encoding="utf-8")
    resolved_path = source_path.resolve()
    document = _document_for(
        resolved_path=resolved_path,
        text=text,
        extension=".md",
        media_type=MARKDOWN_MEDIA_TYPE,
    )
    blocks = _parse_markdown_blocks(text, str(resolved_path))
    metadata = dict(document.metadata)
    metadata["block_count"] = len(blocks)
    return StructuredLoadResult(
        document=document,
        blocks=blocks,
        metadata=metadata,
    )


def load_text_document(path: str | Path) -> StructuredLoadResult:
    source_path = _validate_supported_file(path, TEXT_EXTENSIONS)
    text = source_path.read_text(encoding="utf-8")
    resolved_path = source_path.resolve()
    document = _document_for(
        resolved_path=resolved_path,
        text=text,
        extension=".txt",
        media_type=TEXT_MEDIA_TYPE,
    )
    blocks = _parse_text_blocks(text, str(resolved_path))
    metadata = dict(document.metadata)
    metadata["block_count"] = len(blocks)
    return StructuredLoadResult(
        document=document,
        blocks=blocks,
        metadata=metadata,
    )


def _load_supported_document(path: str | Path) -> StructuredLoadResult:
    source_path = Path(path).expanduser()
    extension = source_path.suffix.lower()
    if extension in MARKDOWN_EXTENSIONS:
        return load_markdown_document(source_path)
    if extension in TEXT_EXTENSIONS:
        return load_text_document(source_path)
    _raise_unsupported(source_path, extension)


def _validate_supported_file(
    path: str | Path,
    supported_extensions: set[str],
) -> Path:
    source_path = Path(path).expanduser()

    if not source_path.exists():
        raise FileNotFoundError(f"Document not found: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"Document path is not a file: {source_path}")

    extension = source_path.suffix.lower()
    if extension not in supported_extensions:
        _raise_unsupported(source_path, extension)
    return source_path


def _raise_unsupported(source_path: Path, extension: str) -> Never:
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(
        f"Unsupported file type '{extension}' for {source_path}. "
        f"Supported file types: {supported}."
    )


def _document_for(
    *,
    resolved_path: Path,
    text: str,
    extension: str,
    media_type: str,
) -> Document:
    metadata = {
        "source_path": str(resolved_path),
        "file_name": resolved_path.name,
        "extension": extension,
        "media_type": media_type,
        "character_count": len(text),
    }
    return Document(id=str(resolved_path), text=text, metadata=metadata)


def _parse_markdown_blocks(
    text: str,
    source_path: str,
) -> tuple[DocumentBlock, ...]:
    lines = _line_records(text)
    blocks: list[DocumentBlock] = []
    heading_stack: list[str | None] = [None] * 6
    index = 0

    while index < len(lines):
        line = lines[index]
        content = line.content
        if not content.strip():
            index += 1
            continue

        fence_match = _FENCE_PATTERN.match(content)
        if fence_match:
            index = _append_code_block(
                blocks=blocks,
                lines=lines,
                start_index=index,
                source_path=source_path,
                fence=fence_match.group(1),
                code_language=fence_match.group(2) or "",
                section_metadata=_section_metadata(heading_stack),
            )
            continue

        heading_match = _ATX_HEADING_PATTERN.match(content.strip())
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack[level - 1] = title
            for position in range(level, len(heading_stack)):
                heading_stack[position] = None
            metadata = {
                "heading_level": level,
                "section_title": title,
                "heading_path": _heading_path(heading_stack),
            }
            blocks.append(
                _make_markdown_block(
                    source_path=source_path,
                    block_index=len(blocks),
                    block_type="heading",
                    text=text,
                    start=line.start,
                    end=line.end,
                    metadata=metadata,
                )
            )
            index += 1
            continue

        if _is_table_start(lines, index):
            index = _append_line_group_block(
                blocks=blocks,
                lines=lines,
                start_index=index,
                source_path=source_path,
                block_type="table",
                should_continue=lambda next_line: (
                    bool(next_line.content.strip())
                    and "|" in next_line.content
                ),
                metadata={
                    **_section_metadata(heading_stack),
                    "serialization": "markdown_table",
                },
            )
            continue

        if _is_blockquote_line(content):
            index = _append_line_group_block(
                blocks=blocks,
                lines=lines,
                start_index=index,
                source_path=source_path,
                block_type="blockquote",
                should_continue=lambda next_line: _is_blockquote_line(
                    next_line.content
                ),
                metadata=_section_metadata(heading_stack),
            )
            continue

        if _is_list_line(content):
            index = _append_line_group_block(
                blocks=blocks,
                lines=lines,
                start_index=index,
                source_path=source_path,
                block_type="list",
                should_continue=lambda next_line: _is_list_line(next_line.content),
                metadata=_section_metadata(heading_stack),
            )
            continue

        index = _append_paragraph_block(
            blocks=blocks,
            lines=lines,
            start_index=index,
            source_path=source_path,
            heading_stack=heading_stack,
        )

    return tuple(blocks)


def _parse_text_blocks(text: str, source_path: str) -> tuple[DocumentBlock, ...]:
    lines = _line_records(text)
    blocks: list[DocumentBlock] = []
    index = 0

    while index < len(lines):
        if not lines[index].content.strip():
            index += 1
            continue

        start_index = index
        while index < len(lines) and lines[index].content.strip():
            index += 1
        start = lines[start_index].start
        end = lines[index - 1].end
        blocks.append(
            _make_block(
                source_path=source_path,
                media_type=TEXT_MEDIA_TYPE,
                block_index=len(blocks),
                block_type="paragraph",
                text=text,
                start=start,
                end=end,
                metadata={},
            )
        )

    return tuple(blocks)


def _append_code_block(
    *,
    blocks: list[DocumentBlock],
    lines: list[_LineRecord],
    start_index: int,
    source_path: str,
    fence: str,
    code_language: str,
    section_metadata: dict[str, Any],
) -> int:
    index = start_index + 1
    while index < len(lines):
        if lines[index].content.lstrip().startswith(fence):
            index += 1
            break
        index += 1
    metadata = dict(section_metadata)
    if code_language:
        metadata["code_language"] = code_language
    blocks.append(
        _make_markdown_block(
            source_path=source_path,
            block_index=len(blocks),
            block_type="code",
            text=_source_text(lines),
            start=lines[start_index].start,
            end=lines[index - 1].end,
            metadata=metadata,
        )
    )
    return index


def _append_line_group_block(
    *,
    blocks: list[DocumentBlock],
    lines: list[_LineRecord],
    start_index: int,
    source_path: str,
    block_type: BlockType,
    should_continue: Callable[[_LineRecord], bool],
    metadata: dict[str, Any],
) -> int:
    index = start_index + 1
    while index < len(lines) and should_continue(lines[index]):
        index += 1
    blocks.append(
        _make_markdown_block(
            source_path=source_path,
            block_index=len(blocks),
            block_type=block_type,
            text=_source_text(lines),
            start=lines[start_index].start,
            end=lines[index - 1].end,
            metadata=metadata,
        )
    )
    return index


def _append_paragraph_block(
    *,
    blocks: list[DocumentBlock],
    lines: list[_LineRecord],
    start_index: int,
    source_path: str,
    heading_stack: list[str | None],
) -> int:
    index = start_index + 1
    while index < len(lines):
        if not lines[index].content.strip() or _is_special_line(lines, index):
            break
        index += 1
    blocks.append(
        _make_markdown_block(
            source_path=source_path,
            block_index=len(blocks),
            block_type="paragraph",
            text=_source_text(lines),
            start=lines[start_index].start,
            end=lines[index - 1].end,
            metadata=_section_metadata(heading_stack),
        )
    )
    return index


def _make_markdown_block(
    *,
    source_path: str,
    block_index: int,
    block_type: BlockType,
    text: str,
    start: int,
    end: int,
    metadata: dict[str, Any],
) -> DocumentBlock:
    return _make_block(
        source_path=source_path,
        media_type=MARKDOWN_MEDIA_TYPE,
        block_index=block_index,
        block_type=block_type,
        text=text,
        start=start,
        end=end,
        metadata=metadata,
    )


def _make_block(
    *,
    source_path: str,
    media_type: str,
    block_index: int,
    block_type: BlockType,
    text: str,
    start: int,
    end: int,
    metadata: dict[str, Any],
) -> DocumentBlock:
    block_text = _block_text(text, start, end)
    block_metadata = {
        "media_type": media_type,
        "start_char": start,
        "end_char": start + len(block_text),
        **metadata,
    }
    return DocumentBlock(
        source_path=source_path,
        media_type=media_type,
        page_number=None,
        block_index=block_index,
        block_type=block_type,
        text=block_text,
        metadata=block_metadata,
    )


def _block_text(text: str, start: int, end: int) -> str:
    return text[start:end].rstrip("\r\n")


def _is_special_line(lines: list[_LineRecord], index: int) -> bool:
    content = lines[index].content
    return (
        _FENCE_PATTERN.match(content) is not None
        or _ATX_HEADING_PATTERN.match(content.strip()) is not None
        or _is_table_start(lines, index)
        or _is_blockquote_line(content)
        or _is_list_line(content)
    )


def _is_table_start(lines: list[_LineRecord], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index].content and _is_table_separator(lines[index + 1].content)


def _is_table_separator(line: str) -> bool:
    stripped = line.strip().strip("|")
    if "|" not in line or not stripped:
        return False
    cells = [cell.strip() for cell in stripped.split("|")]
    return len(cells) >= 2 and all(
        re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
    )


def _is_blockquote_line(line: str) -> bool:
    return line.lstrip().startswith(">")


def _is_list_line(line: str) -> bool:
    return _LIST_ITEM_PATTERN.match(line) is not None


def _section_metadata(heading_stack: list[str | None]) -> dict[str, Any]:
    heading_path = _heading_path(heading_stack)
    section_title = heading_path[-1] if heading_path else None
    return {
        "section_title": section_title,
        "heading_path": heading_path,
    }


def _heading_path(heading_stack: list[str | None]) -> list[str]:
    return [heading for heading in heading_stack if heading]


def _source_text(lines: list[_LineRecord]) -> str:
    if not lines:
        return ""
    first = lines[0].start
    last = lines[-1].end
    # The line records are from one source string, so this slice is only used
    # through offsets in caller-owned text. It is never exposed directly.
    return lines[0].source[first:last]


class _LineRecord:
    def __init__(self, source: str, start: int, end: int) -> None:
        self.source = source
        self.start = start
        self.end = end
        self.raw = source[start:end]
        self.content = self.raw.rstrip("\r\n")


def _line_records(text: str) -> list[_LineRecord]:
    records: list[_LineRecord] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        end = offset + len(raw_line)
        records.append(_LineRecord(text, offset, end))
        offset = end
    if text and not records:
        records.append(_LineRecord(text, 0, len(text)))
    return records
