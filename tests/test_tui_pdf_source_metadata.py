from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.tui.shell_models import (
    ShellState,
    TranscriptSource,
    format_shell_inspector,
    format_transcript_sources,
)
from ragent_forge.tui.view_models import (
    AskPageState,
    SearchPageState,
    format_ask_page,
    format_ask_source_inspector,
    format_search_page,
    format_search_result_inspector,
)


def make_pdf_result() -> SearchResult:
    return SearchResult(
        chunk_id="/knowledge/paper.pdf::chunk-0031",
        document_id="/knowledge/paper.pdf",
        source_path="/knowledge/paper.pdf",
        score=0.8321,
        text="| Method | Hit@1 | MRR |\n|---|---|---|\n| hybrid | 1.00 | 1.00 |",
        metadata={
            "media_type": "application/pdf",
            "page_start": 7,
            "page_end": 7,
            "block_types": ["table"],
            "block_type": "table",
            "table_indices": [2],
            "table_caption": "Table 2: Retrieval Evaluation Results",
            "table_context_strategy": "same_page_caption_before_table",
            "reading_order_strategy": "coordinate_blocks",
            "table_text_dedup_applied": True,
            "possible_formula": True,
            "possible_formula_lines": ["MRR = reciprocal rank mean"],
            "header_footer_filter_applied": True,
            "retrieval_method": "lexical_token_overlap",
            "warnings": [
                {
                    "source_path": "/knowledge/paper.pdf",
                    "page": 7,
                    "kind": "table_malformed",
                    "message": "Inconsistent row widths.",
                }
            ],
        },
    )


def test_tui_search_and_ask_pages_show_pdf_source_labels() -> None:
    result = make_pdf_result()

    search_text = format_search_page(
        SearchPageState(
            query="metrics",
            results=[result],
            selected_result=result,
            has_searched=True,
        )
    )
    ask_text = format_ask_page(
        AskPageState(
            question="metrics",
            status="Generation: not configured. Showing retrieved context only.",
            sources=[result],
            selected_source=result,
            generation_status="not_configured",
            generation_provider="null",
            has_run=True,
        )
    )

    assert "paper.pdf p.7 table 2" in search_text
    assert "paper.pdf p.7 table 2" in ask_text


def test_tui_inspectors_show_pdf_page_table_metadata() -> None:
    result = make_pdf_result()

    search_inspector = format_search_result_inspector(result, "lexical")
    ask_inspector = format_ask_source_inspector(result)

    for text in (search_inspector, ask_inspector):
        assert "source: paper.pdf p.7 table 2" in text
        assert "type: pdf" in text
        assert "page range: 7" in text
        assert "block type: table" in text
        assert "table: 2" in text
        assert "table caption: Table 2: Retrieval Evaluation Results" in text
        assert "reading order: coordinate_blocks" in text
        assert "dedup: applied" in text
        assert "possible formula: yes" in text
        assert "header/footer: filtered" in text
        assert "warnings: table_malformed" in text


def test_shell_source_list_and_inspector_show_pdf_metadata() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="/knowledge/paper.pdf::chunk-0031",
        source_path="/knowledge/paper.pdf",
        score=0.8321,
        preview="| Method | Hit@1 | MRR |",
        metadata=make_pdf_result().metadata,
    )

    sources_text = format_transcript_sources((source,))
    inspector_text = format_shell_inspector(ShellState(selected_source=source))

    assert "paper.pdf p.7 table 2" in sources_text
    assert "source: paper.pdf p.7 table 2" in inspector_text
    assert "type: pdf" in inspector_text
    assert "page range: 7" in inspector_text
    assert "block type: table" in inspector_text
    assert "table: 2" in inspector_text
    assert "table caption: Table 2: Retrieval Evaluation Results" in inspector_text
    assert "possible formula: yes" in inspector_text


def test_tui_inspectors_show_markdown_section_metadata() -> None:
    result = SearchResult(
        chunk_id="/knowledge/rag.md::chunk-0001",
        document_id="/knowledge/rag.md",
        source_path="/knowledge/rag.md",
        score=1.0,
        text="Hybrid retrieval combines lexical and semantic search.",
        metadata={
            "media_type": "text/markdown",
            "block_types": ["paragraph"],
            "section_title": "Hybrid Retrieval",
            "heading_path": ["RAG Basics", "Hybrid Retrieval"],
            "retrieval_method": "lexical_token_overlap",
        },
    )
    source = TranscriptSource(
        rank=1,
        chunk_id=result.chunk_id,
        source_path=result.source_path,
        score=result.score,
        preview=result.text,
        metadata=result.metadata,
    )

    search_inspector = format_search_result_inspector(result, "lexical")
    shell_inspector = format_shell_inspector(ShellState(selected_source=source))

    for text in (search_inspector, shell_inspector):
        assert "source: rag.md" in text
        assert "type: markdown" in text
        assert "section: Hybrid Retrieval" in text
        assert "heading path: RAG Basics > Hybrid Retrieval" in text
        assert "block type: paragraph" in text


def test_tui_inspectors_show_text_source_metadata() -> None:
    result = SearchResult(
        chunk_id="/knowledge/notes.txt::chunk-0000",
        document_id="/knowledge/notes.txt",
        source_path="/knowledge/notes.txt",
        score=1.0,
        text="Plain notes.",
        metadata={
            "media_type": "text/plain",
            "block_types": ["paragraph"],
            "retrieval_method": "lexical_token_overlap",
        },
    )

    inspector_text = format_search_result_inspector(result, "lexical")

    assert "source: notes.txt" in inspector_text
    assert "type: text" in inspector_text
    assert "block type: paragraph" in inspector_text
