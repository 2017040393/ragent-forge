from ragent_forge.app.source_labels import format_source_label, format_source_range


def test_format_source_label_returns_basename_without_metadata() -> None:
    assert format_source_label("/very/long/path/rag.md") == "rag.md"


def test_format_source_label_formats_pdf_page() -> None:
    assert (
        format_source_label(
            "/knowledge/paper.pdf",
            {
                "media_type": "application/pdf",
                "page_start": 7,
                "page_end": 7,
            },
        )
        == "paper.pdf p.7"
    )


def test_format_source_label_formats_pdf_page_range() -> None:
    assert (
        format_source_label(
            "/knowledge/paper.pdf",
            {
                "media_type": "application/pdf",
                "page_start": 7,
                "page_end": 8,
            },
        )
        == "paper.pdf pp.7-8"
    )


def test_format_source_label_formats_single_pdf_table_index() -> None:
    assert (
        format_source_label(
            "/knowledge/paper.pdf",
            {
                "media_type": "application/pdf",
                "page_start": 7,
                "page_end": 7,
                "table_indices": [2],
            },
        )
        == "paper.pdf p.7 table 2"
    )


def test_format_source_label_tolerates_malformed_metadata() -> None:
    assert (
        format_source_label(
            "/knowledge/paper.pdf",
            {
                "media_type": "application/pdf",
                "page_start": "seven",
                "table_indices": ["two", 3],
            },
        )
        == "paper.pdf"
    )


def test_format_source_range_keeps_character_range() -> None:
    assert format_source_range(0, 100, None) == "0-100"


def test_format_source_range_formats_pdf_page() -> None:
    assert (
        format_source_range(
            None,
            None,
            {
                "media_type": "application/pdf",
                "page_start": 7,
                "page_end": 7,
            },
        )
        == "p.7"
    )


def test_format_source_range_formats_pdf_page_range() -> None:
    assert (
        format_source_range(
            None,
            None,
            {
                "media_type": "application/pdf",
                "page_start": 7,
                "page_end": 8,
            },
        )
        == "pp.7-8"
    )


def test_format_source_range_returns_empty_without_range_metadata() -> None:
    assert format_source_range(None, None, {}) == ""
