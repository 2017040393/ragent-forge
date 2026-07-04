from pathlib import Path

import pytest

from ragent_forge.core.ingestion.structured_loader import (
    SUPPORTED_EXTENSIONS,
    load_structured_document,
)


def test_structured_loader_dispatches_markdown_to_blocks(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# RAG\n\nRetrieval augmented generation.", encoding="utf-8")

    result = load_structured_document(source)

    assert result.document.id == str(source.resolve())
    assert result.document.metadata["media_type"] == "text/markdown"
    assert [block.block_type for block in result.blocks] == ["heading", "paragraph"]
    assert {block.media_type for block in result.blocks} == {"text/markdown"}
    assert result.metadata["media_type"] == "text/markdown"


def test_structured_loader_dispatches_text_to_blocks(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    result = load_structured_document(source)

    assert result.document.metadata["media_type"] == "text/plain"
    assert [block.block_type for block in result.blocks] == [
        "paragraph",
        "paragraph",
    ]
    assert {block.media_type for block in result.blocks} == {"text/plain"}
    assert result.metadata["media_type"] == "text/plain"


def test_structured_loader_keeps_single_extension_registry() -> None:
    assert {".md", ".txt", ".pdf"}.issubset(SUPPORTED_EXTENSIONS)


def test_structured_loader_rejects_unsupported_file(tmp_path: Path) -> None:
    source = tmp_path / "notes.html"
    source.write_text("<p>not supported</p>", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_structured_document(source)
