from pathlib import Path

import pytest

from ragent_forge.core.ingestion.markdown_loader import load_document


def test_loads_markdown_document(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# RAG\n\nRetrieval augmented generation.", encoding="utf-8")

    document = load_document(source)

    assert document.id == str(source.resolve())
    assert document.text == "# RAG\n\nRetrieval augmented generation."
    assert document.metadata["source_path"] == str(source.resolve())
    assert document.metadata["file_name"] == "notes.md"
    assert document.metadata["extension"] == ".md"
    assert document.metadata["character_count"] == len(document.text)


def test_loads_txt_document(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Plain text knowledge.", encoding="utf-8")

    document = load_document(source)

    assert document.text == "Plain text knowledge."
    assert document.metadata["extension"] == ".txt"


def test_rejects_unsupported_file_extension(tmp_path: Path) -> None:
    source = tmp_path / "notes.pdf"
    source.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(source)


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    source = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError, match="Document not found"):
        load_document(source)
