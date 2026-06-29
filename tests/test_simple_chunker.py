from ragent_forge.app.models import Document
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def test_chunks_document_by_character_length() -> None:
    document = Document(
        id="/knowledge/rag.md",
        text="abcdefghijklmnopqrstuvwxyz",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunker = SimpleChunker(chunk_size=10, chunk_overlap=0)

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "klmnopqrst", "uvwxyz"]
    assert [chunk.id for chunk in chunks] == [
        "/knowledge/rag.md::chunk-0000",
        "/knowledge/rag.md::chunk-0001",
        "/knowledge/rag.md::chunk-0002",
    ]
    assert chunks[0].metadata["source_path"] == "/knowledge/rag.md"


def test_chunks_document_with_overlap() -> None:
    document = Document(
        id="/knowledge/rag.md",
        text="abcdefghijkl",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunker = SimpleChunker(chunk_size=5, chunk_overlap=2)

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == ["abcde", "defgh", "ghijk", "jkl"]


def test_empty_document_returns_no_chunks() -> None:
    document = Document(id="/knowledge/empty.md", text="", metadata={})
    chunker = SimpleChunker(chunk_size=10, chunk_overlap=0)

    assert chunker.chunk(document) == []


def test_invalid_chunk_size_is_rejected() -> None:
    try:
        SimpleChunker(chunk_size=0)
    except ValueError as exc:
        assert "chunk_size must be greater than 0" in str(exc)
    else:
        raise AssertionError("Expected invalid chunk_size to be rejected")


def test_invalid_chunk_overlap_is_rejected() -> None:
    try:
        SimpleChunker(chunk_size=10, chunk_overlap=10)
    except ValueError as exc:
        assert "chunk_overlap must satisfy" in str(exc)
    else:
        raise AssertionError("Expected invalid chunk_overlap to be rejected")
