from pathlib import Path

from ragent_forge.app.models import (
    Document,
    IngestResult,
    OperationTrace,
    TraceStep,
)
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.tui.view_models import (
    SearchPageState,
    compact_path_label,
    compact_source_label,
    format_chunk_inspector,
    format_documents_page,
    format_search_page,
    format_search_result_inspector,
    format_settings_page,
    format_trace_inspector,
    format_trace_page,
    load_documents_page_model,
    load_settings_page_model,
    load_trace_page_model,
    page_for_key,
    run_tui_search,
)


def make_workspace(tmp_path: Path) -> LocalWorkspace:
    source_path = str(tmp_path / "examples" / "knowledge" / "rag.md")
    document = Document(
        id=source_path,
        text="Agentic RAG adds planning.\nRetrieval augmented generation basics.",
        metadata={"source_path": source_path},
    )
    chunks = SimpleChunker(chunk_size=32, chunk_overlap=0).chunk(document)
    result = IngestResult(
        source_path=str(tmp_path / "examples" / "knowledge"),
        documents=[document],
        chunks=chunks,
        skipped_files=[],
        metadata={"chunk_size": 32, "chunk_overlap": 0},
    )
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(result.chunks)
    workspace.write_ingest_summary(result)
    return workspace


def test_compact_source_label_returns_file_name_for_long_path() -> None:
    assert compact_source_label("/very/long/path/agentic_rag.md") == "agentic_rag.md"


def test_compact_path_label_shortens_long_absolute_path() -> None:
    assert (
        compact_path_label(
            "C:/Users/MKS/Desktop/code/python/ragent-forge/examples/knowledge"
        )
        == ".../examples/knowledge"
    )


def test_compact_path_label_shortens_windows_backslash_path() -> None:
    assert (
        compact_path_label(
            r"C:\Users\MKS\Desktop\code\python\ragent-forge\examples\knowledge"
        )
        == ".../examples/knowledge"
    )


def test_compact_path_label_shortens_posix_absolute_path() -> None:
    assert (
        compact_path_label("/home/user/code/ragent-forge/examples/knowledge")
        == ".../examples/knowledge"
    )


def test_compact_path_label_preserves_short_relative_path() -> None:
    assert compact_path_label(".ragent/chunks/chunks.jsonl", max_parts=3) == (
        ".ragent/chunks/chunks.jsonl"
    )


def test_compact_path_label_normalizes_ragent_backslash_path() -> None:
    assert compact_path_label(r".ragent\chunks\chunks.jsonl", max_parts=3) == (
        ".ragent/chunks/chunks.jsonl"
    )


def test_compact_path_label_preserves_short_file_name() -> None:
    assert compact_path_label("rag.md") == "rag.md"


def test_documents_page_omits_duplicate_title_when_layout_owns_title(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)

    text = format_documents_page(load_documents_page_model(workspace.root_path))

    assert not text.startswith("Documents")


def test_documents_page_model_uses_compact_summary_paths_and_rows(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)

    model = load_documents_page_model(workspace.root_path)
    page_text = format_documents_page(model)
    inspector_text = format_chunk_inspector(model.recent_chunks[0])

    assert "Workspace" in page_text
    assert "  status: ready" in page_text
    assert "Ingest" in page_text
    assert "  source: .../examples/knowledge" in page_text
    assert "Files" in page_text
    assert "  chunks: .ragent/chunks/chunks.jsonl" in page_text
    assert "  summary: .ragent/ingest/latest_summary.json" in page_text
    assert "Semantic index" in page_text
    assert "  status: missing" in page_text
    assert "Recent chunks" in page_text
    assert "rag.md" in page_text
    assert str(workspace.root_path) not in page_text
    assert "Chunk details" in inspector_text
    assert "source: rag.md" in inspector_text
    assert "full source_path:" in inspector_text
    assert str(Path(model.recent_chunks[0].source_path)) in inspector_text


def test_documents_page_missing_chunks_renders_friendly_message(
    tmp_path: Path,
) -> None:
    model = load_documents_page_model(tmp_path / ".ragent")

    text = format_documents_page(model)

    assert "No chunks found." in text
    assert "ragent ingest examples/knowledge --workspace .ragent" in text


def test_settings_page_hides_configured_api_keys(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    workspace.config_path.write_text(
        "\n".join(
            [
                "[generation]",
                'provider = "openai_responses"',
                'base_url = "https://api.example.test/v1"',
                'model = "gpt-test"',
                'api_key = "generation-secret"',
                "",
                "[embedding]",
                'provider = "openai_embeddings"',
                'base_url = "https://api.example.test/v1"',
                'model = "embed-test"',
                'api_key = "embedding-secret"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    text = format_settings_page(load_settings_page_model(workspace.root_path))

    assert "generation api_key: <hidden>" in text
    assert "embedding api_key: <hidden>" in text
    assert "generation-secret" not in text
    assert "embedding-secret" not in text


def test_settings_page_compacts_vector_index_path(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    workspace.vector_index_path.write_text("", encoding="utf-8")

    text = format_settings_page(load_settings_page_model(workspace.root_path))

    assert "vector index status: ready" in text
    assert "vector index path: .ragent/index/vector_index.jsonl" in text
    assert str(workspace.root_path) not in text


def test_settings_page_config_error_does_not_display_api_key(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.ensure_exists()
    workspace.config_path.write_text(
        "\n".join(
            [
                "[generation]",
                'provider = "openai_responses"',
                'base_url = "https://api.example.test/v1"',
                'model = "gpt-test"',
                'api_key = "generation-secret"',
                'temperature = "hot"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    text = format_settings_page(load_settings_page_model(workspace.root_path))

    assert "Config error: unable to load config." in text
    assert "generation-secret" not in text


def test_search_page_defaults_to_lexical_and_limit_five() -> None:
    state = SearchPageState()

    assert state.query == ""
    assert state.retrieval_mode == "lexical"
    assert state.limit == 5
    assert "Enter a query and press Enter or Run Search." in format_search_page(state)


def test_tui_search_rejects_empty_query(tmp_path: Path) -> None:
    state = run_tui_search(tmp_path / ".ragent", "   ", "lexical", 5)

    assert state.error == "Enter a search query."
    assert state.results == []


def test_tui_lexical_search_renders_compact_results_and_inspector(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)

    state = run_tui_search(workspace.root_path, "Agentic", "lexical", 5)
    page_text = format_search_page(state)
    inspector_text = format_search_result_inspector(
        state.selected_result,
        state.retrieval_mode,
    )

    assert state.error is None
    assert len(state.results) == 1
    assert "Results: 1 | mode: lexical | limit: 5" in page_text
    assert "rag.md" in page_text
    assert str(workspace.root_path) not in page_text
    assert "Search result" in inspector_text
    assert "method: lexical_token_overlap" in inspector_text
    assert "RRF:" not in inspector_text
    assert "full source_path:" in inspector_text


def test_tui_semantic_search_missing_vector_index_is_friendly(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)

    state = run_tui_search(workspace.root_path, "Agentic", "semantic", 5)

    assert state.error == "Vector index not found. Run `ragent index build` first."
    assert state.results == []


def test_tui_hybrid_search_missing_vector_index_is_friendly(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)

    state = run_tui_search(workspace.root_path, "Agentic", "hybrid", 5)

    assert state.error == "Vector index not found. Run `ragent index build` first."
    assert state.results == []


def test_search_page_empty_results_message_is_friendly() -> None:
    state = SearchPageState(
        query="missing",
        retrieval_mode="lexical",
        limit=5,
        results=[],
        has_searched=True,
    )

    text = format_search_page(state)

    assert "No matches found. Try another query or retrieval mode." in text


def test_search_result_inspector_includes_hybrid_metadata_when_present() -> None:
    result = SearchResult(
        chunk_id="chunk-0000",
        document_id="doc",
        source_path="/knowledge/rag.md",
        start_char=0,
        end_char=42,
        score=0.0325,
        text="Agentic RAG adds planning.",
        metadata={
            "retrieval_method": "hybrid_rrf",
            "fusion_method": "reciprocal_rank_fusion",
            "matched_modes": ["lexical", "semantic"],
            "lexical_rank": 1,
            "semantic_rank": 2,
        },
    )

    text = format_search_result_inspector(result, "hybrid")

    assert "method: hybrid_rrf" in text
    assert "RRF:" in text
    assert "  fusion: reciprocal_rank_fusion" in text
    assert "  matched: lexical, semantic" in text
    assert "lexical_rank: 1" in text
    assert "semantic_rank: 2" in text


def test_search_result_inspector_omits_hybrid_fields_for_lexical_result() -> None:
    result = SearchResult(
        chunk_id="chunk-0000",
        document_id="doc",
        source_path="/knowledge/rag.md",
        start_char=0,
        end_char=42,
        score=1.0,
        text="Agentic RAG adds planning.",
        metadata={"retrieval_method": "lexical_token_overlap"},
    )

    text = format_search_result_inspector(result, "lexical")

    assert "method: lexical_token_overlap" in text
    assert "RRF:" not in text
    assert "lexical_rank" not in text
    assert "semantic_rank" not in text


def test_trace_page_renders_recent_steps_and_inspector_metadata(
    tmp_path: Path,
) -> None:
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_trace(
        OperationTrace(
            trace_id="search-20260702T000000Z",
            operation="search",
            status="success",
            started_at="2026-07-02T00:00:00Z",
            finished_at="2026-07-02T00:00:01Z",
            steps=[TraceStep(name="read_chunks", description="Read chunks.")],
            metadata={"retrieval_mode": "hybrid", "result_count": 2},
        )
    )

    model = load_trace_page_model(workspace.root_path)
    page_text = format_trace_page(model)
    inspector_text = format_trace_inspector(model.selected_trace)

    assert "Recent traces" in page_text
    assert "search | success" in page_text
    assert "Steps" in page_text
    assert "1. read_chunks" in page_text
    assert "Trace details" in inspector_text
    assert "- retrieval_mode: hybrid" in inspector_text
    assert "- result_count: 2" in inspector_text


def test_trace_missing_state_suggests_trace_producing_commands(
    tmp_path: Path,
) -> None:
    model = load_trace_page_model(tmp_path / ".ragent")

    text = format_trace_page(model)

    assert "No traces found." in text
    assert 'ragent search "agent memory"' in text


def test_settings_missing_config_state_suggests_config_init(
    tmp_path: Path,
) -> None:
    text = format_settings_page(load_settings_page_model(tmp_path / ".ragent"))

    assert "No config file found. Effective defaults are being used." in text
    assert "ragent config init" in text


def test_navigation_keys_switch_to_required_pages() -> None:
    assert page_for_key("d") == "documents"
    assert page_for_key("1") == "documents"
    assert page_for_key("s") == "search"
    assert page_for_key("2") == "search"
    assert page_for_key("t") == "trace"
    assert page_for_key("3") == "trace"
    assert page_for_key("g") == "settings"
    assert page_for_key("4") == "settings"
