from dataclasses import replace
from pathlib import Path

import pytest
from textual import events
from textual.containers import ScrollableContainer
from textual.css.query import NoMatches
from textual.widgets import Footer, Input, Static
from textual.worker import WorkerState

from ragent_forge.app.models import Document, IngestResult, OperationTrace, TraceStep
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.tui import main as tui_main
from ragent_forge.tui.main import RagentForgeApp
from ragent_forge.tui.shell_models import WELCOME_MESSAGE, TranscriptSource
from ragent_forge.tui.view_models import AskPageState, SearchPageState


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


def assert_missing_widget(app: RagentForgeApp, selector: str) -> None:
    with pytest.raises(NoMatches):
        app.query_one(selector)


def make_transcript_source(rank: int = 1) -> TranscriptSource:
    return TranscriptSource(
        rank=rank,
        chunk_id=f"/knowledge/rag.md::chunk-{rank:04d}",
        source_path=f"/knowledge/source_{rank}.md",
        score=0.1 * rank,
        preview=f"Preview {rank}",
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_tui_app_opens_as_single_shell_without_old_pages(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        assert not hasattr(app, "current_page")
        assert "mode: lexical" in str(
            app.query_one("#shell-status", Static).renderable
        )
        assert "RAGentForge command shell." in str(
            app.query_one("#shell-transcript", Static).renderable
        )
        assert app.query_one("#shell-input", Input).placeholder == (
            "Ask a question or type /help"
        )
        assert app.query_one("#shell-suggestions", Static).renderable == ""
        assert app.focused == app.query_one("#shell-input", Input)
        assert "Shell details" in str(
            app.query_one("#inspector-content", Static).renderable
        )

        for selector in (
            "#navigation",
            "#documents-page",
            "#search-page",
            "#ask-page",
            "#trace-page",
            "#settings-page",
            "#query-input",
            "#ask-question-input",
            "#run-search",
            "#run-ask",
            "#help",
        ):
            assert_missing_widget(app, selector)
        with pytest.raises(NoMatches):
            app.query_one(Footer)


def test_tui_app_has_no_global_key_bindings() -> None:
    assert RagentForgeApp.BINDINGS == []


@pytest.mark.anyio
async def test_tui_app_shell_help_and_mode_commands_update_transcript(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)

        shell_input.value = "/help"
        app._submit_shell_input()
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "Slash commands" in transcript
        assert app.focused == shell_input

        shell_input.value = "/mode hybrid"
        app._submit_shell_input()
        assert app.shell_state.retrieval_mode == "hybrid"
        assert "mode: hybrid" in str(
            app.query_one("#shell-status", Static).renderable
        )
        assert app.focused == shell_input


@pytest.mark.anyio
async def test_tui_app_shell_clear_leaves_input_focused(tmp_path: Path) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/help"
        app._submit_shell_input()

        shell_input.value = "/clear"
        app._submit_shell_input()

        assert app.focused == shell_input
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "RAGentForge command shell." in transcript
        assert "Slash commands" not in transcript


@pytest.mark.anyio
async def test_tui_app_shell_running_submission_preserves_input_text(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        app.shell_state = replace(app.shell_state, running=True)
        shell_input.value = "Do not lose this"

        app._submit_shell_input()

        assert shell_input.value == "Do not lose this"
        assert app.shell_state.messages[-1].text == WELCOME_MESSAGE


@pytest.mark.anyio
async def test_tui_app_shell_suggestions_render_for_slash_prefix(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        app._render_shell_suggestions("/se")

        suggestions = str(app.query_one("#shell-suggestions", Static).renderable)
        assert "/search <query>" in suggestions
        assert "/settings" in suggestions


@pytest.mark.anyio
async def test_tui_app_shell_submission_clears_suggestions(tmp_path: Path) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        app._render_shell_suggestions("/se")
        shell_input.value = "/help"

        app._submit_shell_input()

        assert app.query_one("#shell-suggestions", Static).renderable == ""
        assert shell_input.value == ""


@pytest.mark.anyio
async def test_tui_app_shell_source_command_updates_inspector(tmp_path: Path) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    first = make_transcript_source(1)
    second = make_transcript_source(2)

    async with app.run_test():
        app.shell_state = replace(
            app.shell_state,
            available_sources=(first, second),
            selected_source=first,
        )
        app._render_inspector()
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/source 2"

        app._submit_shell_input()

        inspector = str(app.query_one("#inspector-content", Static).renderable)
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "selected source: source_2.md" in inspector
        assert "rank: 2" in inspector
        assert "selected source 2: source_2.md" in transcript


@pytest.mark.anyio
async def test_tui_app_shell_suggestions_can_move_selection(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/se"
        app._render_shell_suggestions()
        assert "> /search <query>" in str(
            app.query_one("#shell-suggestions", Static).renderable
        )

        app._move_shell_suggestion(1)

        suggestions = str(app.query_one("#shell-suggestions", Static).renderable)
        assert "  /search <query>" in suggestions
        assert "> /settings" in suggestions

        app._move_shell_suggestion(-1)

        assert "> /search <query>" in str(
            app.query_one("#shell-suggestions", Static).renderable
        )


@pytest.mark.anyio
async def test_tui_app_shell_tab_completes_selected_suggestion(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/se"
        app._render_shell_suggestions()
        app._move_shell_suggestion(1)

        assert app._complete_shell_suggestion() is True

        assert shell_input.value == "/settings "
        assert shell_input.cursor_position == len("/settings ")
        assert app.query_one("#shell-suggestions", Static).renderable == ""


@pytest.mark.anyio
async def test_tui_app_shell_enter_completes_suggestion_without_executing(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        before_messages = app.shell_state.messages
        shell_input.value = "/se"
        app._render_shell_suggestions()

        app.on_key(events.Key("enter", "\r"))

        assert shell_input.value == "/search "
        assert app.shell_state.messages == before_messages
        assert "Running search" not in str(
            app.query_one("#shell-transcript", Static).renderable
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

    async with app.run_test():
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

    async with app.run_test():
        app._set_shell_running(True)

        event = FakeEvent()
        app._handle_shell_ask_worker_state(event)  # type: ignore[arg-type]

        assert event.stopped is True
        assert app.shell_state.running is False
        assert app.query_one("#shell-input", Input).disabled is False
        assert app.focused == app.query_one("#shell-input", Input)
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

    async with app.run_test():
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


@pytest.mark.anyio
async def test_tui_app_shell_search_worker_success_refocuses_input(
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
        name = "shell-search"
        result = SearchPageState(
            query="Agentic",
            retrieval_mode="lexical",
            results=[source],
            has_searched=True,
        )

    class FakeEvent:
        state = WorkerState.SUCCESS
        worker = FakeWorker()
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    async with app.run_test():
        app._set_shell_running(True)

        event = FakeEvent()
        app._handle_shell_search_worker_state(event)  # type: ignore[arg-type]

        assert event.stopped is True
        assert app.shell_state.running is False
        assert app.query_one("#shell-input", Input).disabled is False
        assert app.focused == app.query_one("#shell-input", Input)
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "Search results for: Agentic" in transcript
        assert "agentic_rag.md" in transcript


@pytest.mark.anyio
async def test_tui_app_shell_render_requests_transcript_scroll_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    scroll_calls: list[bool] = []

    async with app.run_test():
        container = app.query_one("#shell-transcript-container", ScrollableContainer)

        def fake_scroll_end(*, animate: bool = True, **_kwargs: object) -> None:
            scroll_calls.append(animate)

        monkeypatch.setattr(container, "scroll_end", fake_scroll_end)
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/help"

        app._submit_shell_input()

        assert scroll_calls == [False]


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
