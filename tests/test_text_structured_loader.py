from pathlib import Path

from ragent_forge.core.chunking.block_chunker import BlockChunker
from ragent_forge.core.ingestion.markdown_loader import load_text_document


def test_text_structured_loader_splits_blank_line_paragraphs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Alpha text.\n\nBeta text.\nGamma text.", encoding="utf-8")

    result = load_text_document(source)

    assert result.document.metadata["media_type"] == "text/plain"
    assert result.document.metadata["extension"] == ".txt"
    assert [block.block_type for block in result.blocks] == [
        "paragraph",
        "paragraph",
    ]
    assert [block.text for block in result.blocks] == [
        "Alpha text.",
        "Beta text.\nGamma text.",
    ]
    assert {block.metadata["media_type"] for block in result.blocks} == {
        "text/plain"
    }


def test_text_structured_loader_preserves_character_offsets(tmp_path: Path) -> None:
    source = tmp_path / "offsets.txt"
    text = "Alpha text.\n\nBeta text."
    source.write_text(text, encoding="utf-8")

    result = load_text_document(source)

    for block in result.blocks:
        start_char = block.metadata["start_char"]
        end_char = block.metadata["end_char"]
        assert isinstance(start_char, int)
        assert isinstance(end_char, int)
        assert text[start_char:end_char] == block.text


def test_text_long_paragraph_can_be_split_by_block_chunker(tmp_path: Path) -> None:
    source = tmp_path / "long.txt"
    source.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")

    result = load_text_document(source)
    chunks = BlockChunker(chunk_size=10, chunk_overlap=3).chunk(
        result.document,
        result.blocks,
    )

    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]
    assert chunks[0].metadata["start_char"] == 0
    assert chunks[0].metadata["end_char"] == 10
    assert chunks[-1].metadata["start_char"] == 21
    assert chunks[-1].metadata["end_char"] == 26
