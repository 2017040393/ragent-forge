from ragent_forge.core.ingestion.pdf_reading_order import PdfTextLine
from ragent_forge.core.ingestion.pdf_text_filters import (
    filter_repeated_header_footer_lines,
)


def line(text: str) -> PdfTextLine:
    return PdfTextLine(text=text, top=0.0, x0=0.0)


def test_header_footer_filter_keeps_body_occurrence_of_header_candidate() -> None:
    result = filter_repeated_header_footer_lines(
        [
            [
                line("RAGentForge Technical Report"),
                line("Unique body text page 1"),
                line("RAGentForge Technical Report"),
                line("More body text page 1"),
                line("Confidential Draft"),
            ],
            [
                line("RAGentForge Technical Report"),
                line("Unique body text page 2"),
                line("More body text page 2"),
                line("Conclusion body text page 2"),
                line("Confidential Draft"),
            ],
        ]
    )

    assert result.pages[0].text == "\n".join(
        [
            "Unique body text page 1",
            "RAGentForge Technical Report",
            "More body text page 1",
        ]
    )
    assert result.pages[1].text == "\n".join(
        [
            "Unique body text page 2",
            "More body text page 2",
            "Conclusion body text page 2",
        ]
    )
    assert result.suspected_headers_filtered == 2
    assert result.suspected_footers_filtered == 2
