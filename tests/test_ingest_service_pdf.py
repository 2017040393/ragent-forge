from datetime import datetime
from pathlib import Path

from ragent_forge.app.services.ask_service import AskService
from ragent_forge.app.services.ingest_service import IngestService
from ragent_forge.app.services.search_service import LexicalSearchService
from ragent_forge.app.services.trace_service import build_ingest_trace
from ragent_forge.app.workspace import LocalWorkspace
from tests.pdf_test_utils import write_table_pdf, write_text_pdf


def test_ingest_service_discovers_pdf_and_preserves_markdown_behavior(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("markdown still works", encoding="utf-8")
    (knowledge_dir / "skip.bin").write_text("unsupported", encoding="utf-8")
    pdf_path = knowledge_dir / "paper.pdf"
    write_text_pdf(pdf_path, first_page_text="PDF retrieval text layer.")

    result = IngestService(chunk_size=100, chunk_overlap=0).ingest(knowledge_dir)

    assert result.document_count == 2
    assert result.skipped_files == [str((knowledge_dir / "skip.bin").resolve())]
    assert any(document.metadata["extension"] == ".md" for document in result.documents)
    pdf_documents = [
        document
        for document in result.documents
        if document.metadata.get("media_type") == "application/pdf"
    ]
    assert len(pdf_documents) == 1
    pdf_chunks = [
        chunk
        for chunk in result.chunks
        if chunk.metadata.get("media_type") == "application/pdf"
    ]
    assert len(pdf_chunks) == 1
    assert pdf_chunks[0].metadata["page_start"] == 1
    assert pdf_chunks[0].metadata["page_end"] == 1
    assert pdf_chunks[0].metadata["block_types"] == ["paragraph"]
    assert pdf_chunks[0].metadata["extraction_method"] == "pdf_structured"
    assert result.metadata["pdf"]["pdf_files_seen"] == 1
    assert result.metadata["pdf"]["pdf_files_ingested"] == 1
    assert result.metadata["pdf"]["pdf_pages_seen"] == 1
    assert result.metadata["pdf"]["pdf_pages_with_text"] == 1


def test_ingest_service_writes_table_metadata_and_pdf_warnings(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    write_table_pdf(knowledge_dir / "table_report.pdf")
    write_text_pdf(
        knowledge_dir / "blank_warning.pdf",
        first_page_text="First page has text.",
        second_page_blank=True,
    )

    result = IngestService(chunk_size=1000, chunk_overlap=0).ingest(knowledge_dir)

    table_chunks = [
        chunk
        for chunk in result.chunks
        if "table" in chunk.metadata.get("block_types", [])
    ]
    assert len(table_chunks) == 1
    table_chunk = table_chunks[0]
    assert table_chunk.metadata["table_indices"] == [1]
    assert table_chunk.metadata["block_type"] == "table"
    assert "| Method | Hit@1 | MRR |" in table_chunk.text
    assert result.metadata["pdf"]["pdf_files_seen"] == 2
    assert result.metadata["pdf"]["pdf_tables_extracted"] == 1
    assert result.metadata["pdf"]["pdf_empty_pages"] == 1
    assert result.metadata["pdf"]["pdf_warnings"][0]["kind"] == "empty_page"


def test_pdf_ingest_summary_trace_search_and_ask_use_pdf_chunks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper.pdf"
    write_text_pdf(source, first_page_text="PDF answer source alpha beta.")
    workspace = LocalWorkspace(tmp_path / ".ragent")
    result = IngestService(chunk_size=100, chunk_overlap=0).ingest(source)
    chunks_path = workspace.write_chunks(result.chunks)
    summary_path = workspace.write_ingest_summary(result)
    trace = build_ingest_trace(
        result=result,
        chunks_path=chunks_path,
        summary_path=summary_path,
        started_at=datetime(2026, 7, 3),
        finished_at=datetime(2026, 7, 3),
    )

    search_results = LexicalSearchService(workspace).search("alpha")
    ask_result = AskService(workspace).ask("alpha")
    summary = workspace.read_ingest_summary()

    assert summary["metadata"]["pdf"]["pdf_files_seen"] == 1
    assert trace.metadata["pdf"]["pdf_pages_with_text"] == 1
    assert trace.steps[0].outputs["pdf_files_seen"] == 1
    assert len(search_results) == 1
    assert search_results[0].metadata["media_type"] == "application/pdf"
    assert search_results[0].metadata["page_start"] == 1
    assert len(ask_result.sources) == 1
    assert ask_result.sources[0].source_path == str(source.resolve())
