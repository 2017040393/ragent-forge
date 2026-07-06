import hashlib
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from ragent_forge.app.models import Document
from ragent_forge.app.services import evidence_span_service
from ragent_forge.app.services.evidence_span_service import EvidenceSpanService
from ragent_forge.core.ingestion.document_blocks import DocumentBlock
from ragent_forge.core.ingestion.structured_result import StructuredLoadResult


def test_extracts_markdown_spans_with_section_metadata(tmp_path: Path) -> None:
    first_paragraph = (
        "Agentic RAG keeps retrieval evidence visible and inspectable so eval "
        "datasets can point back to stable source text."
    )
    second_paragraph = (
        "Hybrid retrieval combines lexical matches with semantic matches while "
        "still preserving the original document section."
    )
    generation_paragraph = (
        "Generation answers should cite compact context windows rather than "
        "workspace chunk ids created by a previous ingest run."
    )
    text = (
        "# RAG Guide\n\n"
        "## Retrieval\n\n"
        f"{first_paragraph}\n\n"
        f"{second_paragraph}\n\n"
        "## Empty Section\n\n"
        "## Generation\n\n"
        f"{generation_paragraph}\n"
    )
    source = tmp_path / "guide.md"
    source.write_text(text, encoding="utf-8")

    spans = EvidenceSpanService(min_chars=60, max_chars=500).extract(source)

    assert [span.section_title for span in spans] == ["Retrieval", "Generation"]
    first = spans[0]
    assert first.id == f"{source.resolve()}::span-0000"
    assert first.source_path == str(source.resolve())
    assert first.document_id == str(source.resolve())
    assert first.media_type == "text/markdown"
    assert first.heading_path == ("RAG Guide", "Retrieval")
    assert first.block_types == ("paragraph",)
    assert first.start_char == text.index(first_paragraph)
    assert first.end_char == text.index(second_paragraph) + len(second_paragraph)
    assert first.text == f"{first_paragraph}\n\n{second_paragraph}"
    assert first.metadata["block_indices"] == [2, 3]
    assert "## Retrieval" not in first.text
    assert "Empty Section" not in {span.section_title for span in spans}


def test_respects_min_chars_max_chars_max_spans_and_order(
    tmp_path: Path,
) -> None:
    short = "Too short."
    first = "Alpha evidence sentence " * 3
    second = "Beta evidence sentence " * 3
    third = "Gamma evidence sentence " * 3
    source = tmp_path / "ordered.md"
    source.write_text(
        "# Notes\n\n"
        "## Small\n\n"
        f"{short}\n\n"
        "## Long\n\n"
        f"{first}\n\n"
        f"{second}\n\n"
        f"{third}\n",
        encoding="utf-8",
    )
    service = EvidenceSpanService(min_chars=40, max_chars=160)

    spans = service.extract(tmp_path, max_spans=1)
    repeat = service.extract(tmp_path, max_spans=1)

    assert len(spans) == 1
    assert [span.id for span in spans] == [span.id for span in repeat]
    assert spans[0].section_title == "Long"
    assert first.strip() in spans[0].text
    assert second.strip() in spans[0].text
    assert third.strip() not in spans[0].text
    assert len(spans[0].text) <= 160
    assert short not in spans[0].text


def test_extracts_txt_paragraph_spans(tmp_path: Path) -> None:
    first = (
        "Plain text evidence can still become a stable span even without any "
        "Markdown heading metadata."
    )
    second = (
        "Blank lines create paragraph blocks, and consecutive useful paragraph "
        "blocks should be grouped deterministically."
    )
    source = tmp_path / "notes.txt"
    source.write_text(f"{first}\n\n{second}\n", encoding="utf-8")

    spans = EvidenceSpanService(min_chars=50, max_chars=500).extract(source)

    assert len(spans) == 1
    span = spans[0]
    assert span.media_type == "text/plain"
    assert span.section_title is None
    assert span.heading_path == ()
    assert span.block_types == ("paragraph",)
    assert span.text == f"{first}\n\n{second}"


def test_extract_ignores_workspace_chunks_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text(
        "# Source\n\n"
        "Real evidence paragraph that should be extracted from the source tree.",
        encoding="utf-8",
    )
    chunks_dir = tmp_path / ".ragent" / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunk.md").write_text(
        "# Chunk\n\nThis stale workspace chunk must not become evidence.",
        encoding="utf-8",
    )

    spans = EvidenceSpanService(min_chars=20).extract(tmp_path)

    assert len(spans) == 1
    assert "Real evidence paragraph" in spans[0].text
    assert "stale workspace chunk" not in spans[0].text


def test_pdf_is_skipped_by_default_and_attempted_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = tmp_path / "guide.md"
    markdown.write_text(
        "# Guide\n\n"
        "Markdown evidence should be extracted while colocated PDFs stay gated.",
        encoding="utf-8",
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nnot a real pdf")
    real_load_structured_document = evidence_span_service.load_structured_document
    loaded_file_names: list[str] = []

    def recording_load_structured_document(path: str | Path):
        path = Path(path)
        loaded_file_names.append(path.name)
        if path.suffix.lower() == ".pdf":
            raise RuntimeError("PDF loading attempted")
        return real_load_structured_document(path)

    monkeypatch.setattr(
        evidence_span_service,
        "load_structured_document",
        recording_load_structured_document,
    )

    spans = EvidenceSpanService(min_chars=20).extract(tmp_path)

    assert len(spans) == 1
    assert loaded_file_names == ["guide.md"]
    assert "colocated PDFs stay gated" in spans[0].text

    loaded_file_names.clear()
    with pytest.raises(RuntimeError, match="PDF loading attempted"):
        EvidenceSpanService(min_chars=20, include_pdf=True).extract(tmp_path)

    assert loaded_file_names == ["guide.md", "paper.pdf"]


def test_include_pdf_extracts_page_aware_spans_from_generated_pdf(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "two_pages.pdf"
    _write_two_page_pdf(pdf)

    spans = EvidenceSpanService(
        min_chars=20,
        max_chars=1000,
        include_pdf=True,
    ).extract(pdf)

    assert len(spans) == 2
    assert [span.page_start for span in spans] == [1, 2]
    assert [span.page_end for span in spans] == [1, 2]
    assert all(span.media_type == "application/pdf" for span in spans)
    assert all(span.start_char is None for span in spans)
    assert all(span.end_char is None for span in spans)
    assert all(span.metadata["offsets_available"] is False for span in spans)
    assert all("text_sha256" in span.metadata for span in spans)
    assert spans[0].metadata["page_numbers"] == [1]
    assert spans[1].metadata["page_numbers"] == [2]
    assert spans[0].metadata["extraction_method"] == "pdfplumber"
    assert "First page evidence" in spans[0].text
    assert "Second page evidence" not in spans[0].text
    assert "Second page evidence" in spans[1].text
    assert spans[0].metadata["text_sha256"] == hashlib.sha256(
        spans[0].text.encode("utf-8")
    ).hexdigest()


def test_pdf_table_blocks_keep_table_spans_standalone_and_metadata_rich(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "tables.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake")
    resolved_path = str(pdf.resolve())
    document = Document(
        id=resolved_path,
        text="Intro paragraph\n\n| Metric | Value |\n|---|---|\n| Recall | 0.8 |",
        metadata={
            "source_path": resolved_path,
            "media_type": "application/pdf",
        },
    )
    paragraph = DocumentBlock(
        source_path=resolved_path,
        media_type="application/pdf",
        page_number=3,
        block_index=0,
        block_type="paragraph",
        text="Intro paragraph for the table page.",
        metadata={
            "page_number": 3,
            "media_type": "application/pdf",
            "extraction_method": "pdfplumber",
            "reading_order_strategy": "coordinate_blocks",
            "reading_order_warning": "minor overlap",
            "possible_formula": True,
            "possible_formula_lines": ["E = mc^2"],
            "table_text_dedup_applied": True,
            "table_text_dedup_strategy": "line_exact_match",
            "table_text_dedup_removed_lines": 2,
            "header_footer_filter_applied": True,
            "header_footer_removed_lines": 1,
            "header_footer_candidates": ["Confidential"],
        },
    )
    table = DocumentBlock(
        source_path=resolved_path,
        media_type="application/pdf",
        page_number=3,
        block_index=1,
        block_type="table",
        text="| Metric | Value |\n|---|---|\n| Recall | 0.8 |",
        metadata={
            "page_number": 3,
            "media_type": "application/pdf",
            "table_index": 2,
            "table_caption": "Table 2: Retrieval quality",
            "table_context": "Metrics section",
            "table_context_strategy": "same_page_caption_before_table",
            "row_count": 2,
            "column_count": 2,
            "serialization": "markdown_table",
            "extraction_method": "pdfplumber",
            "warnings": [
                {
                    "source_path": resolved_path,
                    "page": 3,
                    "kind": "table_malformed",
                    "message": "Extracted table had inconsistent row widths.",
                }
            ],
        },
    )

    def fake_load_structured_document(path: str | Path) -> StructuredLoadResult:
        assert Path(path) == pdf
        return StructuredLoadResult(
            document=document,
            blocks=(paragraph, table),
            metadata=document.metadata,
        )

    monkeypatch.setattr(
        evidence_span_service,
        "load_structured_document",
        fake_load_structured_document,
    )

    spans = EvidenceSpanService(
        min_chars=10,
        max_chars=1000,
        include_pdf=True,
    ).extract(pdf)

    assert [span.block_types for span in spans] == [("paragraph",), ("table",)]
    table_span = spans[1]
    assert table_span.page_start == 3
    assert table_span.page_end == 3
    assert table_span.metadata["page_numbers"] == [3]
    assert table_span.metadata["table_indices"] == [2]
    assert table_span.metadata["table_caption"] == "Table 2: Retrieval quality"
    assert table_span.metadata["table_context"] == "Metrics section"
    assert (
        table_span.metadata["table_context_strategy"]
        == "same_page_caption_before_table"
    )
    assert table_span.metadata["row_count"] == 2
    assert table_span.metadata["column_count"] == 2
    assert table_span.metadata["serialization"] == "markdown_table"
    assert table_span.metadata["warnings"][0]["kind"] == "table_malformed"
    assert table_span.metadata["offsets_available"] is False
    paragraph_span = spans[0]
    assert paragraph_span.metadata["reading_order_warning"] == "minor overlap"
    assert paragraph_span.metadata["possible_formula"] is True
    assert paragraph_span.metadata["possible_formula_lines"] == ["E = mc^2"]
    assert paragraph_span.metadata["table_text_dedup_applied"] is True
    assert paragraph_span.metadata["table_text_dedup_removed_lines"] == 2
    assert paragraph_span.metadata["header_footer_filter_applied"] is True
    assert paragraph_span.metadata["header_footer_removed_lines"] == 1
    assert paragraph_span.metadata["header_footer_candidates"] == ["Confidential"]


def test_short_standalone_table_span_is_kept_but_short_paragraph_is_filtered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "short_table.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake")
    resolved_path = str(pdf.resolve())
    document = Document(
        id=resolved_path,
        text="tiny note\n\n| A |\n|---|\n| 1 |",
        metadata={
            "source_path": resolved_path,
            "media_type": "application/pdf",
        },
    )
    short_paragraph = DocumentBlock(
        source_path=resolved_path,
        media_type="application/pdf",
        page_number=1,
        block_index=0,
        block_type="paragraph",
        text="tiny note",
        metadata={
            "page_number": 1,
            "media_type": "application/pdf",
            "extraction_method": "pdfplumber",
        },
    )
    short_table = DocumentBlock(
        source_path=resolved_path,
        media_type="application/pdf",
        page_number=1,
        block_index=1,
        block_type="table",
        text="| A |\n|---|\n| 1 |",
        metadata={
            "page_number": 1,
            "media_type": "application/pdf",
            "table_index": 1,
            "row_count": 1,
            "column_count": 1,
            "serialization": "markdown_table",
            "extraction_method": "pdfplumber",
        },
    )

    def fake_load_structured_document(path: str | Path) -> StructuredLoadResult:
        assert Path(path) == pdf
        return StructuredLoadResult(
            document=document,
            blocks=(short_paragraph, short_table),
            metadata=document.metadata,
        )

    monkeypatch.setattr(
        evidence_span_service,
        "load_structured_document",
        fake_load_structured_document,
    )

    spans = EvidenceSpanService(
        min_chars=80,
        max_chars=1000,
        include_pdf=True,
    ).extract(pdf)

    assert len(spans) == 1
    span = spans[0]
    assert span.block_types == ("table",)
    assert span.text == "| A |\n|---|\n| 1 |"
    assert span.metadata["below_min_chars_allowed"] is True
    assert span.metadata["table_indices"] == [1]
    assert "tiny note" not in span.text


def test_extract_raises_clear_errors_for_missing_or_unsupported_paths(
    tmp_path: Path,
) -> None:
    service = EvidenceSpanService()

    with pytest.raises(FileNotFoundError, match="Evidence source path not found"):
        service.extract(tmp_path / "missing")

    unsupported = tmp_path / "image.bin"
    unsupported.write_bytes(b"unsupported")

    with pytest.raises(ValueError, match="No supported Markdown/TXT files found"):
        service.extract(unsupported)


def _write_two_page_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(
        72,
        740,
        "First page evidence explains lexical retrieval and local inspection.",
    )
    pdf.showPage()
    pdf.drawString(
        72,
        740,
        "Second page evidence explains semantic retrieval and page boundaries.",
    )
    pdf.save()
