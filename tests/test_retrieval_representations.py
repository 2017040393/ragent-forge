from ragent_forge.core.retrieval.contracts import ChunkRecord
from ragent_forge.core.retrieval.representations import (
    build_embedding_text,
    build_structured_document_text_v1,
)


def make_chunk(metadata: dict[str, object]) -> ChunkRecord:
    return {
        "schema_version": 2,
        "snapshot_id": None,
        "chunk_id": "docs/guide.md::chunk-0000",
        "document_id": "docs/guide.md",
        "text": "The original evidence stays unchanged.",
        "source_path": "docs/guide.md",
        "start_char": 0,
        "end_char": 37,
        "metadata": metadata,
        "source_kind": "document",
        "provenance": None,
        "authority": "source",
        "freshness": None,
        "lifecycle": "regenerable",
    }


def test_structured_document_text_v1_has_stable_field_order() -> None:
    chunk = make_chunk(
        {
            "heading_path": ["Retrieval", "Hybrid"],
            "section_title": "Hybrid",
            "block_types": ["paragraph", "table", "paragraph"],
            "page_start": 4,
            "page_end": 5,
            "possible_formula": True,
        }
    )

    assert build_structured_document_text_v1(chunk) == (
        "Document title: Retrieval\n"
        "Source: docs/guide.md\n"
        "Section: Retrieval > Hybrid\n"
        "Page: 4-5\n"
        "Block types: paragraph, table\n"
        "Signals: formula, table\n"
        "Content:\n"
        "The original evidence stays unchanged."
    )


def test_structured_document_text_v1_uses_pdf_fallbacks() -> None:
    chunk = make_chunk(
        {
            "block_type": "paragraph",
            "media_type": "application/pdf",
            "page_start": 9,
        }
    )

    text = build_structured_document_text_v1(chunk)

    assert "Document title: guide" in text
    assert "Section: unknown" in text
    assert "Page: 9" in text
    assert "Block types: paragraph" in text
    assert "Signals: none" in text


def test_raw_representation_is_exactly_chunk_text() -> None:
    chunk = make_chunk({"heading_path": ["Ignored for E0"]})

    assert build_embedding_text(chunk, "raw_chunk_text_v1") == chunk["text"]
