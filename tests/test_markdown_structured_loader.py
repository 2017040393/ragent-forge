from pathlib import Path

from ragent_forge.core.ingestion.markdown_loader import load_markdown_document


def test_markdown_structured_loader_detects_headings_and_section_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rag.md"
    source.write_text(
        "# RAG Basics\n\n## Hybrid Retrieval\n\nCombine lexical and semantic search.",
        encoding="utf-8",
    )

    result = load_markdown_document(source)

    assert result.document.metadata["media_type"] == "text/markdown"
    assert [block.block_type for block in result.blocks] == [
        "heading",
        "heading",
        "paragraph",
    ]
    paragraph = result.blocks[-1]
    assert paragraph.metadata["section_title"] == "Hybrid Retrieval"
    assert paragraph.metadata["heading_path"] == [
        "RAG Basics",
        "Hybrid Retrieval",
    ]


def test_markdown_structured_loader_detects_table_and_keeps_markdown_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "table.md"
    source.write_text(
        "# Metrics\n\n"
        "| Method | Strength |\n"
        "|---|---|\n"
        "| lexical | exact terms |\n"
        "| semantic | meaning |\n",
        encoding="utf-8",
    )

    result = load_markdown_document(source)

    table = next(block for block in result.blocks if block.block_type == "table")
    assert table.text == (
        "| Method | Strength |\n"
        "|---|---|\n"
        "| lexical | exact terms |\n"
        "| semantic | meaning |"
    )
    assert table.metadata["serialization"] == "markdown_table"
    assert table.metadata["section_title"] == "Metrics"
    assert table.metadata["heading_path"] == ["Metrics"]


def test_markdown_structured_loader_detects_code_list_and_blockquote(
    tmp_path: Path,
) -> None:
    source = tmp_path / "blocks.md"
    source.write_text(
        "```python\nprint('hi')\n```\n\n"
        "- first\n- second\n\n"
        "> quoted\n> text\n",
        encoding="utf-8",
    )

    result = load_markdown_document(source)

    assert [block.block_type for block in result.blocks] == [
        "code",
        "list",
        "blockquote",
    ]
    assert result.blocks[0].metadata["code_language"] == "python"


def test_markdown_structured_loader_preserves_character_offsets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "offsets.md"
    text = "# Title\n\nParagraph text."
    source.write_text(text, encoding="utf-8")

    result = load_markdown_document(source)

    for block in result.blocks:
        start_char = block.metadata["start_char"]
        end_char = block.metadata["end_char"]
        assert isinstance(start_char, int)
        assert isinstance(end_char, int)
        assert text[start_char:end_char] == block.text
