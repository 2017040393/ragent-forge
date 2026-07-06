from pathlib import Path

import pytest

from ragent_forge.app.services.evidence_span_service import EvidenceSpanService


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


def test_extract_raises_clear_errors_for_missing_or_unsupported_paths(
    tmp_path: Path,
) -> None:
    service = EvidenceSpanService()

    with pytest.raises(FileNotFoundError, match="Evidence source path not found"):
        service.extract(tmp_path / "missing")

    unsupported = tmp_path / "image.bin"
    unsupported.write_bytes(b"unsupported")

    with pytest.raises(ValueError, match="No supported Markdown/TXT/PDF files found"):
        service.extract(unsupported)
