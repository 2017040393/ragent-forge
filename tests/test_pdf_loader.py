from pathlib import Path

import pytest

from ragent_forge.core.ingestion.pdf_loader import load_pdf_document
from tests.pdf_test_utils import (
    write_captioned_table_pdf,
    write_formula_pdf,
    write_header_footer_pdf,
    write_table_pdf,
    write_text_pdf,
    write_two_column_pdf,
)


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


def test_load_pdf_document_orders_two_column_text_by_column(tmp_path: Path) -> None:
    source = tmp_path / "two_column.pdf"
    write_two_column_pdf(source)

    result = load_pdf_document(source)

    paragraph = next(
        block for block in result.blocks if block.block_type == "paragraph"
    )
    assert paragraph.metadata["reading_order_strategy"] == "coordinate_blocks"
    assert paragraph.text.index("Left column beta") < paragraph.text.index(
        "Right column one"
    )
    assert result.metadata["reading_order_strategy"] == "coordinate_blocks"
    assert result.metadata["reading_order_fallback_pages"] == 0


def test_load_pdf_document_adds_table_caption_and_deduplicates_page_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "captioned_table.pdf"
    write_captioned_table_pdf(source)

    result = load_pdf_document(source)

    paragraph = next(
        block for block in result.blocks if block.block_type == "paragraph"
    )
    table = next(block for block in result.blocks if block.block_type == "table")
    assert table.text.startswith("Table 2: Retrieval Evaluation Results\n\n")
    assert table.metadata["table_caption"] == "Table 2: Retrieval Evaluation Results"
    assert table.metadata["table_context_strategy"] == "same_page_caption_before_table"
    assert "lexical 0.67 0.78" not in paragraph.text
    assert "hybrid 1.00 1.00" not in paragraph.text
    assert paragraph.metadata["table_text_dedup_applied"] is True
    assert paragraph.metadata["table_text_dedup_removed_lines"] >= 2
    assert result.metadata["table_text_dedup_pages"] == 1
    assert result.metadata["table_text_dedup_removed_lines"] >= 2


def test_load_pdf_document_preserves_and_marks_formula_like_lines(
    tmp_path: Path,
) -> None:
    source = tmp_path / "formula.pdf"
    write_formula_pdf(source)

    result = load_pdf_document(source)

    paragraph = next(
        block for block in result.blocks if block.block_type == "paragraph"
    )
    assert "RRF(d) = SUM 1 / (k + rank_i(d))" in paragraph.text
    assert paragraph.metadata["possible_formula"] is True
    assert "RRF(d) = SUM 1 / (k + rank_i(d))" in paragraph.metadata[
        "possible_formula_lines"
    ]
    assert result.metadata["possible_formula_blocks"] == 1
    assert result.metadata["possible_formula_lines"] == 2


def test_load_pdf_document_filters_repeated_header_footer_lines(
    tmp_path: Path,
) -> None:
    source = tmp_path / "header_footer.pdf"
    write_header_footer_pdf(source)

    result = load_pdf_document(source)

    paragraphs = [block for block in result.blocks if block.block_type == "paragraph"]
    assert len(paragraphs) == 3
    assert all("RAGentForge Technical Report" not in block.text for block in paragraphs)
    assert all("Confidential Draft" not in block.text for block in paragraphs)
    assert all("Unique body text page" in block.text for block in paragraphs)
    assert all(
        block.metadata["header_footer_filter_applied"] is True
        for block in paragraphs
    )
    assert result.metadata["suspected_headers_filtered"] == 3
    assert result.metadata["suspected_footers_filtered"] == 3


def test_load_pdf_document_rejects_non_pdf_file(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported PDF file type"):
        load_pdf_document(source)
