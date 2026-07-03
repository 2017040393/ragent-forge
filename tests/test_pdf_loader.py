from pathlib import Path

import pytest

from ragent_forge.core.ingestion.pdf_loader import load_pdf_document
from tests.pdf_test_utils import write_table_pdf, write_text_pdf


def test_load_pdf_document_extracts_page_text_and_empty_page_warning(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper.pdf"
    write_text_pdf(
        source,
        first_page_text="RAGentForge reads selectable PDF text.",
        second_page_blank=True,
    )

    result = load_pdf_document(source)

    assert result.document.id == str(source.resolve())
    assert result.document.metadata["media_type"] == "application/pdf"
    assert result.metadata["page_count"] == 2
    assert result.metadata["pages_with_text"] == 1
    assert result.metadata["empty_pages"] == 1
    assert result.metadata["character_count"] > 0
    assert [block.block_type for block in result.blocks] == ["paragraph"]
    assert result.blocks[0].page_number == 1
    assert "selectable PDF text" in result.blocks[0].text
    assert result.warnings[0].kind == "empty_page"
    assert result.warnings[0].page == 2
    assert result.warnings[0].to_dict()["kind"] == "empty_page"


def test_load_pdf_document_extracts_page_local_tables(tmp_path: Path) -> None:
    source = tmp_path / "table_report.pdf"
    write_table_pdf(source)

    result = load_pdf_document(source)

    table_blocks = [block for block in result.blocks if block.block_type == "table"]
    assert len(table_blocks) == 1
    table = table_blocks[0]
    assert table.page_number == 1
    assert table.metadata["table_index"] == 1
    assert table.metadata["row_count"] == 3
    assert table.metadata["column_count"] == 3
    assert table.metadata["serialization"] == "markdown_table"
    assert table.metadata["media_type"] == "application/pdf"
    assert "| Method | Hit@1 | MRR |" in table.text
    assert "| hybrid | 1.00 | 1.00 |" in table.text
    assert result.metadata["tables_extracted"] == 1


def test_load_pdf_document_rejects_non_pdf_file(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported PDF file type"):
        load_pdf_document(source)
