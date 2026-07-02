from dataclasses import replace
from pathlib import Path

import pytest
from textual.containers import ScrollableContainer
from textual.widgets import Button, Input, Static
from textual.worker import WorkerState

from ragent_forge.app.models import Document, IngestResult, OperationTrace, TraceStep
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.tui import main as tui_main
from ragent_forge.tui.main import RagentForgeApp
from ragent_forge.tui.view_models import AskPageState


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
        assert "Search result" in str(inspector.renderable)
        assert "full source_path:" in str(inspector.renderable)
        assert "/knowledge/agentic_rag.md" in str(inspector.renderable)
        assert "/knowledge/agentic_rag.md |" not in str(
            app.query_one("#search-message", Static).renderable
        )


@pytest.mark.anyio
async def test_tui_app_shell_page_renders_status_transcript_and_input(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test() as pilot:
        await pilot.press("h")

        assert app.current_page == "shell"
        assert "mode: lexical" in str(
            app.query_one("#shell-status", Static).renderable
        )
        assert "RAGentForge command shell." in str(
            app.query_one("#shell-transcript", Static).renderable
        )
        assert app.query_one("#shell-input", Input).placeholder == (
            "Ask a question or type /help"
        )


@pytest.mark.anyio
async def test_tui_app_shell_submission_starts_ask_worker_and_uses_shell_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    started_workers: list[tuple[object, dict[str, object]]] = []
    calls: list[tuple[object, str, str, int, int, bool]] = []

    def fake_run_tui_ask(
        workspace_path: object,
        question: str,
        mode: str,
        limit: int,
        max_context_chars: int,
        show_prompt: bool,
    ) -> AskPageState:
        calls.append(
            (workspace_path, question, mode, limit, max_context_chars, show_prompt)
        )
        return AskPageState(question=question, answer="answer", has_run=True)

    def fake_run_worker(work: object, **kwargs: object) -> object:
        started_workers.append((work, kwargs))
        return object()

    monkeypatch.setattr(tui_main, "run_tui_ask", fake_run_tui_ask)
    monkeypatch.setattr(app, "run_worker", fake_run_worker)

    async with app.run_test() as pilot:
        await pilot.press("h")
        app.shell_state = replace(
            app.shell_state,
            retrieval_mode="hybrid",
            limit=3,
            max_context_chars=3000,
            show_prompt=True,
        )
        app._render_shell()
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "What is Agentic RAG?"

        app._submit_shell_input()

        assert calls == []
        assert started_workers
        assert started_workers[0][1]["name"] == "shell-ask"
        assert started_workers[0][1]["group"] == "shell"
        assert started_workers[0][1]["thread"] is True
        assert started_workers[0][1]["exclusive"] is True
        assert shell_input.disabled is True
        assert app.shell_state.running is True
        assert shell_input.value == ""
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "User:\n  What is Agentic RAG?" in transcript
        assert "Running ask for: What is Agentic RAG?" in transcript
        assert "status: running" in str(
            app.query_one("#shell-status", Static).renderable
        )

        work = started_workers[0][0]
        assert callable(work)
        result = work()

        assert isinstance(result, AskPageState)
        assert calls == [
            (
                workspace.root_path,
                "What is Agentic RAG?",
                "hybrid",
                3,
                3000,
                True,
            )
        ]


@pytest.mark.anyio
async def test_tui_app_shell_ask_worker_success_appends_messages_and_selects_source(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    source = SearchResult(
        chunk_id="/knowledge/agentic_rag.md::chunk-0000",
        document_id="/knowledge/agentic_rag.md",
        source_path="/knowledge/agentic_rag.md",
        start_char=0,
        end_char=42,
        score=1.0,
        text="Agentic RAG adds planning before retrieval.",
        metadata={"retrieval_method": "lexical_token_overlap"},
    )

    class FakeWorker:
        name = "shell-ask"
        result = AskPageState(
            question="What is Agentic RAG?",
            retrieval_mode="lexical",
            answer="Agentic RAG adds planning.",
            sources=[source],
            generation_status="success",
            generation_provider="openai_responses",
            has_run=True,
        )

    class FakeEvent:
        state = WorkerState.SUCCESS
        worker = FakeWorker()
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    async with app.run_test() as pilot:
        await pilot.press("h")
        app._set_shell_running(True)

        event = FakeEvent()
        app._handle_shell_ask_worker_state(event)  # type: ignore[arg-type]

        assert event.stopped is True
        assert app.shell_state.running is False
        assert app.query_one("#shell-input", Input).disabled is False
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "Assistant:\n  Agentic RAG adds planning." in transcript
        assert "Sources:" in transcript
        assert "agentic_rag.md" in transcript
        assert app.shell_state.selected_source is not None
        assert app.shell_state.selected_source.source_path == (
            "/knowledge/agentic_rag.md"
        )


@pytest.mark.anyio
async def test_tui_app_shell_search_starts_worker_and_disables_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    started_workers: list[tuple[object, dict[str, object]]] = []

    def forbidden_sync_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Shell search should run inside a worker")

    def fake_run_worker(work: object, **kwargs: object) -> object:
        started_workers.append((work, kwargs))
        return object()

    monkeypatch.setattr(tui_main, "run_tui_search", forbidden_sync_call)
    monkeypatch.setattr(app, "run_worker", fake_run_worker)

    async with app.run_test() as pilot:
        await pilot.press("h")
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/search Agentic"

        app._submit_shell_input()

        assert started_workers
        assert started_workers[0][1]["name"] == "shell-search"
        assert started_workers[0][1]["group"] == "shell"
        assert started_workers[0][1]["thread"] is True
        assert started_workers[0][1]["exclusive"] is True
        assert shell_input.disabled is True
        assert app.shell_state.running is True
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "Running search for: Agentic" in transcript
        assert "status: running" in str(
            app.query_one("#shell-status", Static).renderable
        )


def test_tui_app_shell_read_only_handlers_return_workspace_summaries(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    add_trace(workspace)
    app = RagentForgeApp(workspace.root_path)

    handlers = app._shell_read_only_handlers()

    assert handlers.docs is not None
    assert handlers.trace is not None
    assert handlers.settings is not None
    assert "Workspace" in handlers.docs()
    assert "Latest trace" in handlers.trace()
    assert "read_chunks" in handlers.trace()
    assert "config path:" in handlers.settings()


@pytest.mark.anyio
async def test_tui_app_shell_inspector_shows_basic_details(tmp_path: Path) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test() as pilot:
        await pilot.press("h")

        inspector = str(app.query_one("#inspector-content", Static).renderable)
        assert "Shell details" in inspector
        assert "mode: lexical" in inspector
        assert "messages: 1" in inspector


@pytest.mark.anyio
async def test_tui_app_navigates_and_runs_lexical_ask(tmp_path: Path) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test() as pilot:
        await pilot.press("a")

        assert app.current_page == "ask"

        await pilot.click("#ask-question-input")
        await pilot.press("A", "g", "e", "n", "t", "i", "c")
        await pilot.press("enter")
        await pilot.pause(0.2)

        inspector = app.query_one("#inspector-content", Static)
        assert "Ask source" in str(inspector.renderable)
        assert "full source_path:" in str(inspector.renderable)
        assert "/knowledge/agentic_rag.md" in str(inspector.renderable)


@pytest.mark.anyio
async def test_tui_app_ask_running_state_disables_run_button(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test() as pilot:
        await pilot.press("a")

        app._set_ask_running(True)
        assert app.ask_running is True
        assert app.query_one("#run-ask", Button).disabled is True

        app._set_ask_running(False)
        assert app.ask_running is False
        assert app.query_one("#run-ask", Button).disabled is False


@pytest.mark.anyio
async def test_tui_app_run_ask_starts_worker_without_sync_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    started_workers: list[tuple[object, dict[str, object]]] = []

    def forbidden_sync_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("run_tui_ask should run inside a worker")

    def fake_run_worker(work: object, **kwargs: object) -> object:
        started_workers.append((work, kwargs))
        return object()

    monkeypatch.setattr(tui_main, "run_tui_ask", forbidden_sync_call)
    monkeypatch.setattr(app, "run_worker", fake_run_worker)

    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.click("#ask-question-input")
        await pilot.press("A", "g", "e", "n", "t", "i", "c")
        await pilot.press("enter")

        assert started_workers
        assert started_workers[0][1]["thread"] is True
        assert started_workers[0][1]["exclusive"] is True
        assert app.ask_running is True
        assert app.query_one("#run-ask", Button).disabled is True
        assert "Running ask..." in str(
            app.query_one("#ask-message", Static).renderable
        )


@pytest.mark.anyio
async def test_tui_app_run_ask_ignores_duplicate_start_while_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    started_workers: list[object] = []

    def fake_run_worker(work: object, **_kwargs: object) -> object:
        started_workers.append(work)
        return object()

    monkeypatch.setattr(app, "run_worker", fake_run_worker)

    async with app.run_test() as pilot:
        await pilot.press("a")
        app.ask_running = True

        app._run_ask_from_inputs()

        assert started_workers == []


@pytest.mark.anyio
async def test_tui_app_ask_answer_uses_scroll_container(tmp_path: Path) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test() as pilot:
        await pilot.press("a")

        app.query_one("#ask-answer-container", ScrollableContainer)
        assert "#ask-answer-container" in RagentForgeApp.CSS
        assert "height: 12" in RagentForgeApp.CSS
        assert "#ask-sources-table" in RagentForgeApp.CSS


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
