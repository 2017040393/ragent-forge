from pathlib import Path

import pytest

from ragent_forge.app.services.ingest_service import IngestService


def test_ingests_markdown_and_txt_files_recursively(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    nested_dir = knowledge_dir / "nested"
    nested_dir.mkdir(parents=True)
    (knowledge_dir / "rag.md").write_text("abcdefghij", encoding="utf-8")
    (nested_dir / "notes.txt").write_text("klmnopqrstuv", encoding="utf-8")
    (knowledge_dir / "ignore.bin").write_text("ignored", encoding="utf-8")

    service = IngestService(chunk_size=5, chunk_overlap=1)

    result = service.ingest(knowledge_dir)

    assert result.source_path == str(knowledge_dir.resolve())
    assert result.document_count == 2
    assert result.chunk_count == 6
    assert result.skipped_count == 1
    assert result.skipped_files == [str((knowledge_dir / "ignore.bin").resolve())]
    assert [document.metadata["file_name"] for document in result.documents] == [
        "rag.md",
        "notes.txt",
    ]
    assert [chunk.text for chunk in result.chunks] == [
        "abcde",
        "efghi",
        "ij",
        "klmno",
        "opqrs",
        "stuv",
    ]


def test_ingests_single_supported_file(tmp_path: Path) -> None:
    source = tmp_path / "single.md"
    source.write_text("local first", encoding="utf-8")

    result = IngestService(chunk_size=20, chunk_overlap=0).ingest(source)

    assert result.document_count == 1
    assert result.chunk_count == 1
    assert result.skipped_count == 0
    assert result.documents[0].metadata["file_name"] == "single.md"


def test_ingest_service_allows_custom_chunk_size_without_overlap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "small.md"
    source.write_text("abcdefghij", encoding="utf-8")

    result = IngestService(chunk_size=5).ingest(source)

    assert result.chunk_count == 2
    assert [chunk.text for chunk in result.chunks] == ["abcde", "fghij"]


def test_ingest_rejects_missing_path(tmp_path: Path) -> None:
    service = IngestService()

    with pytest.raises(FileNotFoundError, match="Ingest path not found"):
        service.ingest(tmp_path / "missing")


def test_ingest_rejects_path_with_no_supported_files(tmp_path: Path) -> None:
    (tmp_path / "only.bin").write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="No supported Markdown/TXT/PDF files found"):
        IngestService().ingest(tmp_path)
