from ragent_forge.app.models import Document
from ragent_forge.core.chunking.block_chunker import BlockChunker
from ragent_forge.core.ingestion.document_blocks import DocumentBlock


def test_block_chunker_combines_paragraph_blocks_with_page_metadata() -> None:
    document = Document(
        id="/knowledge/paper.pdf",
        text="",
        metadata={"source_path": "/knowledge/paper.pdf"},
    )
    blocks = (
        DocumentBlock(
            source_path="/knowledge/paper.pdf",
            media_type="application/pdf",
            page_number=1,
            block_index=0,
            block_type="paragraph",
            text="First page text.",
            metadata={"extraction_method": "pdfplumber"},
        ),
        DocumentBlock(
            source_path="/knowledge/paper.pdf",
            media_type="application/pdf",
            page_number=2,
            block_index=1,
            block_type="paragraph",
            text="Second page text.",
            metadata={"extraction_method": "pdfplumber"},
        ),
    )

    chunks = BlockChunker(chunk_size=100, chunk_overlap=0).chunk(document, blocks)

    assert len(chunks) == 1
    assert chunks[0].id == "/knowledge/paper.pdf::chunk-0000"
    assert chunks[0].text == "First page text.\n\nSecond page text."
    assert chunks[0].metadata["media_type"] == "application/pdf"
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["page_end"] == 2
    assert chunks[0].metadata["block_types"] == ["paragraph"]
    assert chunks[0].metadata["extraction_method"] == "pdf_structured"


def test_block_chunker_keeps_table_blocks_standalone() -> None:
    document = Document(
        id="/knowledge/paper.pdf",
        text="",
        metadata={"source_path": "/knowledge/paper.pdf"},
    )
    blocks = (
        DocumentBlock(
            source_path="/knowledge/paper.pdf",
            media_type="application/pdf",
            page_number=7,
            block_index=0,
            block_type="paragraph",
            text="Retrieval evaluation results:",
            metadata={"extraction_method": "pdfplumber"},
        ),
        DocumentBlock(
            source_path="/knowledge/paper.pdf",
            media_type="application/pdf",
            page_number=7,
            block_index=1,
            block_type="table",
            text="| Method | MRR |\n|---|---|\n| hybrid | 1.00 |",
            metadata={
                "table_index": 2,
                "row_count": 2,
                "column_count": 2,
                "warnings": [
                    {
                        "source_path": "/knowledge/paper.pdf",
                        "page": 7,
                        "kind": "table_malformed",
                        "message": "Inconsistent row widths.",
                    }
                ],
            },
        ),
    )

    chunks = BlockChunker(chunk_size=100, chunk_overlap=0).chunk(document, blocks)

    assert [chunk.metadata["block_types"] for chunk in chunks] == [
        ["paragraph"],
        ["table"],
    ]
    assert chunks[1].text.startswith("| Method | MRR |")
    assert chunks[1].metadata["page_start"] == 7
    assert chunks[1].metadata["page_end"] == 7
    assert chunks[1].metadata["table_indices"] == [2]
    assert chunks[1].metadata["block_type"] == "table"
    assert chunks[1].metadata["warnings"][0]["kind"] == "table_malformed"


def test_block_chunker_splits_oversized_non_table_block() -> None:
    document = Document(
        id="/knowledge/paper.pdf",
        text="",
        metadata={"source_path": "/knowledge/paper.pdf"},
    )
    blocks = (
        DocumentBlock(
            source_path="/knowledge/paper.pdf",
            media_type="application/pdf",
            page_number=7,
            block_index=0,
            block_type="paragraph",
            text="abcdefghijklmnopqrstuvwxyz",
            metadata={"extraction_method": "pdfplumber"},
        ),
    )

    chunks = BlockChunker(chunk_size=10, chunk_overlap=3).chunk(document, blocks)

    assert [chunk.id for chunk in chunks] == [
        "/knowledge/paper.pdf::chunk-0000",
        "/knowledge/paper.pdf::chunk-0001",
        "/knowledge/paper.pdf::chunk-0002",
        "/knowledge/paper.pdf::chunk-0003",
    ]
    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[1].text[-3:] == chunks[2].text[:3]
    assert chunks[2].text[-3:] == chunks[3].text[:3]
    for chunk in chunks:
        assert chunk.metadata["media_type"] == "application/pdf"
        assert chunk.metadata["page_start"] == 7
        assert chunk.metadata["page_end"] == 7
        assert chunk.metadata["block_types"] == ["paragraph"]
        assert chunk.metadata["extraction_method"] == "pdf_structured"


def test_block_chunker_keeps_oversized_table_block_standalone() -> None:
    document = Document(
        id="/knowledge/paper.pdf",
        text="",
        metadata={"source_path": "/knowledge/paper.pdf"},
    )
    blocks = (
        DocumentBlock(
            source_path="/knowledge/paper.pdf",
            media_type="application/pdf",
            page_number=3,
            block_index=0,
            block_type="table",
            text="| " + ("cell | " * 20),
            metadata={"table_index": 1},
        ),
    )

    chunks = BlockChunker(chunk_size=20, chunk_overlap=5).chunk(document, blocks)

    assert len(chunks) == 1
    assert chunks[0].metadata["block_types"] == ["table"]
    assert chunks[0].metadata["table_indices"] == [1]
    assert chunks[0].metadata["exceeds_chunk_size"] is True
