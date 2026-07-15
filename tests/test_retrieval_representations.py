from typing import cast

from ragent_forge.core.retrieval.contracts import ChunkRecord
from ragent_forge.core.retrieval.representations import (
    build_embedding_text,
    build_embedding_texts,
    build_query_embedding_text,
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


def test_instructed_query_v1_has_exact_template() -> None:
    query = "What is reciprocal rank fusion?"

    assert build_query_embedding_text(query, "instructed_query_v1") == (
        "Instruct: Retrieve the document passage that best answers the query. "
        "Distinguish the correct section from other passages in the same source "
        "document.\n"
        "Query: What is reciprocal rank fusion?"
    )


def test_raw_query_representation_is_exactly_the_original_query() -> None:
    query = "  preserve surrounding whitespace  "

    assert build_query_embedding_text(query, "raw_query_v1") == query


def test_cleaned_pdf_section_v1_cleans_and_propagates_section() -> None:
    first = cast(
        ChunkRecord,
        {
            **make_chunk(
                {
                    "media_type": "application/pdf",
                    "page_start": 53,
                    "page_end": 53,
                    "block_types": ["paragraph"],
                }
            ),
            "chunk_id": "docs/book.pdf::chunk-0000",
            "document_id": "docs/book.pdf",
            "source_path": "docs/book.pdf",
            "text": (
                "3\nRandom Vectors in High Dimensions\nProb-\nability (cid:30)tools."
            ),
        },
    )
    second = cast(
        ChunkRecord,
        {
            **first,
            "chunk_id": "docs/book.pdf::chunk-0001",
            "text": "54\nFurther results in this chapter.",
        },
    )
    other_document = cast(
        ChunkRecord,
        {
            **first,
            "chunk_id": "docs/other.pdf::chunk-0000",
            "document_id": "docs/other.pdf",
            "source_path": "docs/other.pdf",
            "text": "54\nplain continuation without a heading.",
        },
    )

    texts = build_embedding_texts(
        [first, second, other_document],
        "cleaned_pdf_section_text_v1",
    )

    assert "Section: Random Vectors in High Dimensions" in texts[0]
    assert "Content:\nRandom Vectors in High Dimensions Probability tools." in texts[0]
    assert "Section: Random Vectors in High Dimensions" in texts[1]
    assert "Content:\nFurther results in this chapter." in texts[1]
    assert "Section: unknown" in texts[2]


def test_cleaned_pdf_section_v1_prefers_local_numbered_heading() -> None:
    chunk = cast(
        ChunkRecord,
        {
            **make_chunk(
                {
                    "media_type": "application/pdf",
                    "page_start": 63,
                    "block_types": ["paragraph"],
                }
            ),
            "chunk_id": "docs/book.pdf::chunk-0002",
            "document_id": "docs/book.pdf",
            "source_path": "docs/book.pdf",
            "text": "3.3.2 Multivariate Normal\nThe standard normal distribution...",
        },
    )

    text = build_embedding_text(chunk, "cleaned_pdf_section_text_v1")

    assert "Section: 3.3.2 Multivariate Normal" in text


def test_cleaned_pdf_section_v1_keeps_markdown_representation_unchanged() -> None:
    chunk = make_chunk({"heading_path": ["Retrieval", "Hybrid"]})

    assert build_embedding_text(
        chunk, "cleaned_pdf_section_text_v1"
    ) == build_structured_document_text_v1(chunk)


def test_cleaned_pdf_formula_v1_adds_only_high_confidence_evidence() -> None:
    chunk = cast(
        ChunkRecord,
        {
            **make_chunk(
                {
                    "media_type": "application/pdf",
                    "page_start": 238,
                    "block_types": ["paragraph"],
                    "possible_formula": True,
                    "possible_formula_lines": [
                        "(cid:2)",
                        "\u222b |sin x \u2212 u(x)|2 dx \u2265 0.",
                        "This is ordinary prose without math.",
                        "The proof continues in Section 2.",
                        "MRR is the reciprocal rank mean.",
                        "x \u2208 R2",
                    ],
                }
            ),
            "chunk_id": "docs/book.pdf::chunk-0003",
            "document_id": "docs/book.pdf",
            "source_path": "docs/book.pdf",
            "text": "6.63 example: using linear algebra to approximate sine",
        },
    )

    text = build_embedding_text(chunk, "cleaned_pdf_formula_text_v1")

    assert (
        "Formula evidence: integral |sin x - u(x)|2 dx >= 0. | "
        "x element_of R2"
    ) in text
    formula_line = text.split("Formula evidence: ", 1)[1].splitlines()[0]
    assert "ordinary prose" not in formula_line
    assert "continues in Section" not in formula_line
    assert "reciprocal rank mean" not in formula_line


def test_cleaned_pdf_formula_v1_marks_missing_evidence_explicitly() -> None:
    chunk = cast(
        ChunkRecord,
        {
            **make_chunk(
                {
                    "media_type": "application/pdf",
                    "page_start": 10,
                    "block_types": ["paragraph"],
                }
            ),
            "chunk_id": "docs/book.pdf::chunk-0004",
            "document_id": "docs/book.pdf",
            "source_path": "docs/book.pdf",
            "text": "Plain PDF content.",
        },
    )

    text = build_embedding_text(chunk, "cleaned_pdf_formula_text_v1")

    assert "Formula evidence: none" in text


def test_cleaned_pdf_formula_v1_keeps_markdown_representation_unchanged() -> None:
    chunk = make_chunk({"heading_path": ["Retrieval", "Hybrid"]})

    assert build_embedding_text(
        chunk, "cleaned_pdf_formula_text_v1"
    ) == build_structured_document_text_v1(chunk)
