from pathlib import Path

import pytest
from textual.widgets import Static

from ragent_forge.app.models import Document, IngestResult, OperationTrace, TraceStep
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.tui.main import RagentForgeApp


def make_tui_workspace(tmp_path: Path) -> LocalWorkspace:
    document = Document(
        id="/knowledge/agentic_rag.md",
        text="Agentic RAG adds planning before retrieval.",
        metadata={"source_path": "/knowledge/agentic_rag.md"},
    )
    chunks = SimpleChunker(chunk_size=80, chunk_overlap=0).chunk(document)
    result = IngestResult(
        source_path="/knowledge",
        documents=[document],
        chunks=chunks,
        skipped_files=[],
        metadata={"chunk_size": 80, "chunk_overlap": 0},
    )
    workspace = LocalWorkspace(tmp_path / ".ragent")
    workspace.write_chunks(result.chunks)
    workspace.write_ingest_summary(result)
    return workspace


def add_trace(workspace: LocalWorkspace) -> None:
    workspace.write_trace(
        OperationTrace(
            trace_id="search-20260702T000000Z",
            operation="search",
            status="success",
            started_at="2026-07-02T00:00:00Z",
            finished_at="2026-07-02T00:00:01Z",
            steps=[TraceStep(name="read_chunks", description="Read chunks.")],
            metadata={"retrieval_mode": "lexical"},
        )
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_tui_app_navigates_and_runs_lexical_search(tmp_path: Path) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test() as pilot:
        await pilot.press("s")
        await pilot.click("#query-input")
        await pilot.press("A", "g", "e", "n", "t", "i", "c")
        await pilot.press("enter")

        assert app.current_page == "search"
        inspector = app.query_one("#inspector-content", Static)
        assert "Search result details" in str(inspector.renderable)
        assert "source_path: /knowledge/agentic_rag.md" in str(inspector.renderable)
        assert "/knowledge/agentic_rag.md |" not in str(
            app.query_one("#search-message", Static).renderable
        )


@pytest.mark.anyio
async def test_tui_app_trace_page_shows_selected_trace_steps(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    add_trace(workspace)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test() as pilot:
        await pilot.press("t")

        assert app.current_page == "trace"
        trace_steps = app.query_one("#trace-steps", Static)
        assert "Steps" in str(trace_steps.renderable)
        assert "1. read_chunks" in str(trace_steps.renderable)
