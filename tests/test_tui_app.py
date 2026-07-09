import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import events
from textual.containers import ScrollableContainer
from textual.css.query import NoMatches
from textual.widgets import Footer, Input, Static
from textual.worker import WorkerState

from ragent_forge.app.models import Document, IngestResult, OperationTrace, TraceStep
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.services.session_service import (
    SessionService,
    TuiSessionRun,
    TuiSessionSource,
)
from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.tui import main as tui_main
from ragent_forge.tui.main import RagentForgeApp, _source_picker_label
from ragent_forge.tui.shell_models import TranscriptSource
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


def make_session_source(rank: int = 1) -> TuiSessionSource:
    return TuiSessionSource(
        rank=rank,
        chunk_id=f"/knowledge/rag.md::chunk-{rank:04d}",
        source_path=f"/knowledge/source_{rank}.md",
        score=0.1 * rank,
        preview=f"Preview {rank}",
    )


def test_source_picker_label_includes_location_method_score_and_chunk() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="/knowledge/paper.pdf::chunk-0001",
        source_path="/knowledge/paper.pdf",
        score=0.42,
        preview="PDF evidence",
        metadata={
            "media_type": "application/pdf",
            "page_start": 2,
            "page_end": 2,
            "retrieval_method": "hybrid_rrf",
        },
    )

    assert _source_picker_label(source) == (
        "1. paper.pdf p.2  method=hybrid_rrf  score=0.42  chunk=chunk-0001"
    )


def make_session_run() -> TuiSessionRun:
    return TuiSessionRun(
        retrieval_mode="hybrid",
        retrieval_method="hybrid_rrf",
        limit=5,
        max_context_chars=4000,
        show_prompt=False,
        generation_status="success",
        generation_provider="openai_responses",
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
        assert "mode: hybrid" in str(
            app.query_one("#shell-status", Static).renderable
        )
        assert str(app.query_one("#shell-transcript", Static).renderable) == ""
        assert app.query_one("#shell-input", Input).placeholder == (
            "Ask your knowledge base...  / for commands"
        )
        assert app.query_one("#shell-suggestions", Static).renderable == ""
        assert app.focused == app.query_one("#shell-input", Input)
        assert "No source selected." in str(
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


@pytest.mark.anyio
async def test_tui_app_restores_latest_session_on_mount(tmp_path: Path) -> None:
    workspace = make_tui_workspace(tmp_path)
    service = SessionService(workspace.root_path)
    session = service.create_session()
    service.append_turn(
        session.id,
        question="What is saved?",
        assistant_text="A persisted answer.",
        sources=[make_session_source()],
        run=make_session_run(),
    )
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert app.shell_state.current_session_id == session.id
        assert app.shell_state.current_session_title == "What is saved?"
        assert "User:\n  What is saved?" in transcript
        assert "Assistant: [1 source]\n  A persisted answer." in transcript


def test_tui_app_has_no_global_key_bindings() -> None:
    assert RagentForgeApp.BINDINGS == []


@pytest.mark.parametrize(
    "module_name",
    [
        "ragent_forge.tui.screens.ask",
        "ragent_forge.tui.screens.settings",
        "ragent_forge.tui.widgets.answer_panel",
        "ragent_forge.tui.widgets.source_list",
        "ragent_forge.tui.widgets.trace_view",
    ],
)
def test_tui_does_not_ship_legacy_placeholder_modules(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.anyio
async def test_tui_app_shell_help_opens_modal_and_mode_stays_out_of_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    opened_help: list[bool] = []

    def fake_show_help_modal() -> None:
        opened_help.append(True)

    monkeypatch.setattr(app, "_show_help_modal", fake_show_help_modal)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)

        shell_input.value = "/help"
        app._submit_shell_input()
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert opened_help == [True]
        assert "Slash commands" not in transcript
        assert app.focused == shell_input

        shell_input.value = "/mode bm25"
        app._submit_shell_input()
        assert app.shell_state.retrieval_mode == "bm25"
        assert "mode: bm25" in str(
            app.query_one("#shell-status", Static).renderable
        )
        assert "retrieval mode set to bm25" not in str(
            app.query_one("#shell-transcript", Static).renderable
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
        assert transcript == ""
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
        assert app.shell_state.messages == ()
        assert app.shell_state.notice == (
            "1 draft queued. Press Enter after the current request finishes to send."
        )
        assert "1 draft queued" in str(
            app.query_one("#shell-status", Static).renderable
        )


@pytest.mark.anyio
async def test_tui_app_shell_running_draft_is_ready_after_completion(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        app.shell_state = replace(app.shell_state, running=True)
        shell_input.value = "Queued question"
        app._submit_shell_input()

        app._set_shell_running(False)
        app._render_shell()

        assert shell_input.value == "Queued question"
        assert app.shell_state.notice == "1 draft ready. Press Enter to send."
        assert "1 draft ready" in str(
            app.query_one("#shell-status", Static).renderable
        )


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
        assert "source: source_2.md" in inspector
        assert "rank: 2" in inspector
        assert "selected source 2: source_2.md" not in transcript


@pytest.mark.anyio
async def test_tui_app_shell_sources_command_opens_source_picker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    first = make_transcript_source(1)
    second = make_transcript_source(2)
    opened_sources: list[tuple[TranscriptSource, ...]] = []

    def fake_show_sources_modal() -> None:
        opened_sources.append(app.shell_state.available_sources)

    monkeypatch.setattr(app, "_show_sources_modal", fake_show_sources_modal)

    async with app.run_test():
        app.shell_state = replace(
            app.shell_state,
            available_sources=(first, second),
            selected_source=first,
        )
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/sources"

        app._submit_shell_input()

        assert opened_sources == [(first, second)]
        assert str(app.query_one("#shell-transcript", Static).renderable) == ""


@pytest.mark.anyio
async def test_tui_app_source_picker_selection_updates_inspector(
    tmp_path: Path,
) -> None:
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

        app._handle_source_picker_result(second)

        inspector = str(app.query_one("#inspector-content", Static).renderable)
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert app.shell_state.selected_source == second
        assert "source: source_2.md" in inspector
        assert "rank: 2" in inspector
        assert transcript == ""


@pytest.mark.anyio
async def test_tui_app_session_commands_manage_current_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    service = SessionService(workspace.root_path)
    session = service.create_session()
    saved_session, turn = service.append_turn(
        session.id,
        question="Original question?",
        assistant_text="Original answer.",
        sources=[make_session_source()],
        run=make_session_run(),
    )
    app = RagentForgeApp(workspace.root_path)
    reruns: list[str] = []

    def fake_rerun(question: str) -> None:
        reruns.append(question)

    monkeypatch.setattr(app, "_run_shell_ask_from_dispatch", fake_rerun)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        assert app.shell_state.current_session_id == saved_session.id

        shell_input.value = "/rename Better title"
        app._submit_shell_input()
        assert app.shell_state.current_session_title == "Better title"

        shell_input.value = "/pin"
        app._submit_shell_input()
        assert app.shell_state.session_pinned is True

        shell_input.value = "/star"
        app._submit_shell_input()
        assert app.shell_state.session_starred is True

        shell_input.value = "/export markdown"
        app._submit_shell_input()
        assert "Exported session:" in str(
            app.query_one("#shell-status", Static).renderable
        )

        shell_input.value = "/continue-sources"
        app._submit_shell_input()
        assert shell_input.value.startswith("Using the selected sources, ")

        shell_input.value = "/rerun"
        app._submit_shell_input()
        assert reruns == ["Original question?"]

        shell_input.value = "/branch"
        app._submit_shell_input()
        assert app.shell_state.current_session_id != saved_session.id
        branch = SessionService(workspace.root_path).load_latest_or_create()
        assert branch.branched_from_session_id == saved_session.id
        assert branch.branched_from_turn_id == turn.id


@pytest.mark.anyio
async def test_tui_app_switch_and_session_search_open_session_modal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    service = SessionService(workspace.root_path)
    first = service.create_session("First chat")
    second = service.create_session("Second chat")
    app = RagentForgeApp(workspace.root_path)
    opened: list[list[str]] = []

    def fake_show_sessions_modal() -> None:
        opened.append([summary.title for summary in app.shell_state.session_summaries])

    monkeypatch.setattr(app, "_show_sessions_modal", fake_show_sessions_modal)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        assert app.shell_state.current_session_id == second.id

        shell_input.value = f"/switch {first.id}"
        app._submit_shell_input()
        assert app.shell_state.current_session_id == first.id

        shell_input.value = "/sessions"
        app._submit_shell_input()
        assert opened[-1] == ["Second chat", "First chat"]

        service.set_pinned(first.id, True)
        shell_input.value = "/sessions pinned"
        app._submit_shell_input()
        assert opened[-1] == ["First chat"]

        shell_input.value = "/session-search second"
        app._submit_shell_input()
        assert opened[-1] == ["Second chat"]


@pytest.mark.anyio
async def test_tui_app_delete_requires_second_confirmation(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    service = SessionService(workspace.root_path)
    session = service.create_session("Important chat")
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        assert app.shell_state.current_session_id == session.id

        shell_input.value = "/delete"
        app._submit_shell_input()

        assert service.load_session(session.id).id == session.id
        assert "Type /delete again" in str(
            app.query_one("#shell-status", Static).renderable
        )

        shell_input.value = "/delete"
        app._submit_shell_input()

        with pytest.raises(ValueError, match="Session not found"):
            service.load_session(session.id)


@pytest.mark.anyio
async def test_tui_app_session_picker_enter_switches_and_refocuses_input(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    service = SessionService(workspace.root_path)
    first = service.create_session("First chat")
    second = service.create_session("Second chat")
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test() as pilot:
        shell_input = app.query_one("#shell-input", Input)
        assert app.shell_state.current_session_id == second.id

        shell_input.value = "/sessions"
        app._submit_shell_input()
        await pilot.pause()

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        assert app.shell_state.current_session_id == first.id
        assert app.focused == shell_input


@pytest.mark.anyio
async def test_tui_app_shell_docs_command_uses_inspector_for_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    opened_modals: list[tuple[str, str]] = []

    def fake_show_command_result_modal(title: str, text: str) -> None:
        opened_modals.append((title, text))

    monkeypatch.setattr(
        app,
        "_show_command_result_modal",
        fake_show_command_result_modal,
        raising=False,
    )

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/docs"

        app._submit_shell_input()

        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        inspector = str(app.query_one("#inspector-content", Static).renderable)
        assert "Document summary loaded in Inspector." not in transcript
        assert "Workspace\n" not in transcript
        assert "Workspace\n" not in transcript
        assert "Workspace" in inspector
        assert "Ingest" in inspector
        assert opened_modals
        assert opened_modals[-1][0] == "Documents"
        assert "Workspace" in opened_modals[-1][1]
        assert "Ingest" in opened_modals[-1][1]


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
async def test_tui_app_shell_suggestions_can_move_beyond_visible_window(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/"
        app._render_shell_suggestions()

        for _ in range(8):
            assert app._move_shell_suggestion(1) is True

        suggestions = str(app.query_one("#shell-suggestions", Static).renderable)
        assert "/ask <question>" not in suggestions
        assert "> /new" in suggestions


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
async def test_tui_app_shell_tab_completes_argument_suggestion(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "/mode b"
        app._render_shell_suggestions()

        suggestions = str(app.query_one("#shell-suggestions", Static).renderable)
        assert "> bm25" in suggestions

        assert app._complete_shell_suggestion() is True

        assert shell_input.value == "/mode bm25"
        assert shell_input.cursor_position == len("/mode bm25")
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

    def forbidden_run_tui_ask(*_args: object, **_kwargs: object) -> AskPageState:
        raise AssertionError("TUI ask should use stream_tui_ask")

    def fake_stream_tui_ask(
        workspace_path: object,
        question: str,
        mode: str,
        limit: int,
        max_context_chars: int,
        show_prompt: bool,
    ):
        calls.append(
            (workspace_path, question, mode, limit, max_context_chars, show_prompt)
        )
        yield SimpleNamespace(
            type="done",
            text="",
            state=AskPageState(question=question, answer="answer", has_run=True),
        )

    def fake_run_worker(work: object, **kwargs: object) -> object:
        started_workers.append((work, kwargs))
        return object()

    monkeypatch.setattr(tui_main, "run_tui_ask", forbidden_run_tui_ask)
    monkeypatch.setattr(tui_main, "stream_tui_ask", fake_stream_tui_ask)
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
        assert shell_input.disabled is False
        assert app.focused == shell_input
        assert app.shell_state.running is True
        assert shell_input.value == ""
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "User:\n  What is Agentic RAG?" in transcript
        assert "Assistant:" in transcript
        assert "Running ask for: What is Agentic RAG?" not in transcript

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
async def test_tui_app_shell_ask_worker_streams_answer_into_current_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        metadata={"retrieval_method": "hybrid_rrf"},
    )
    started_workers: list[tuple[object, dict[str, object]]] = []
    snapshots: list[str] = []

    def forbidden_run_tui_ask(*_args: object, **_kwargs: object) -> AskPageState:
        raise AssertionError("TUI ask should use stream_tui_ask")

    def fake_stream_tui_ask(
        workspace_path: object,
        question: str,
        mode: str,
        limit: int,
        max_context_chars: int,
        show_prompt: bool,
    ):
        assert workspace_path == workspace.root_path
        assert question == "What is Agentic RAG?"
        assert mode == "hybrid"
        assert limit == 5
        assert max_context_chars == 4000
        assert show_prompt is False
        yield SimpleNamespace(type="delta", text="Agentic ", state=None)
        snapshots.append(str(app.query_one("#shell-transcript", Static).renderable))
        yield SimpleNamespace(type="delta", text="RAG", state=None)
        snapshots.append(str(app.query_one("#shell-transcript", Static).renderable))
        yield SimpleNamespace(
            type="done",
            text="",
            state=AskPageState(
                question=question,
                retrieval_mode="hybrid",
                answer="Agentic RAG",
                sources=[source],
                generation_status="success",
                generation_provider="openai_responses",
                has_run=True,
            ),
        )

    def fake_run_worker(work: object, **kwargs: object) -> object:
        started_workers.append((work, kwargs))
        return object()

    def immediate_call_from_thread(callback, *args):
        callback(*args)

    monkeypatch.setattr(tui_main, "run_tui_ask", forbidden_run_tui_ask)
    monkeypatch.setattr(tui_main, "stream_tui_ask", fake_stream_tui_ask, raising=False)
    monkeypatch.setattr(app, "run_worker", fake_run_worker)
    monkeypatch.setattr(app, "call_from_thread", immediate_call_from_thread)

    async with app.run_test():
        shell_input = app.query_one("#shell-input", Input)
        shell_input.value = "What is Agentic RAG?"

        app._submit_shell_input()

        work = started_workers[0][0]
        assert callable(work)
        result = work()

        assert isinstance(result, AskPageState)
        assert snapshots[0].endswith("Assistant:\n  Agentic ")
        assert snapshots[1].endswith("Assistant:\n  Agentic RAG")
        final_result = result

        class FakeWorker:
            name = "shell-ask"
            result = final_result

        class FakeEvent:
            state = WorkerState.SUCCESS
            worker = FakeWorker()
            stopped = False

            def stop(self) -> None:
                self.stopped = True

        event = FakeEvent()
        app._handle_shell_ask_worker_state(event)  # type: ignore[arg-type]

        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert transcript.count("Assistant:") == 1
        assert "Assistant: [1 source]\n  Agentic RAG" in transcript
        assert "agentic_rag.md" not in transcript
        assert app.shell_state.selected_source is not None
        assert app.shell_state.selected_turn_id is not None

        saved = SessionService(workspace.root_path).load_latest_or_create()
        assert saved.turns[-1].user_message.text == "What is Agentic RAG?"
        assert saved.turns[-1].assistant_message.text == "Agentic RAG"
        assert saved.turns[-1].sources[0].source_path == "/knowledge/agentic_rag.md"
        assert saved.turns[-1].run is not None
        assert saved.turns[-1].run.retrieval_mode == "hybrid"


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
        assert "Assistant: [1 source]\n  Agentic RAG adds planning." in transcript
        assert "Sources:" not in transcript
        assert "agentic_rag.md" not in transcript
        assert app.shell_state.selected_source is not None
        assert app.shell_state.selected_source.source_path == (
            "/knowledge/agentic_rag.md"
        )


@pytest.mark.anyio
async def test_tui_app_shell_ask_worker_shows_prompt_preview_in_inspector(
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
    prompt_preview = (
        "System: answer with evidence.\n\n"
        "Retrieved context:\n"
        "Agentic RAG adds planning before retrieval."
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
            prompt_preview=prompt_preview,
            show_prompt=True,
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

        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        inspector = str(app.query_one("#inspector-content", Static).renderable)
        assert "Assistant: [1 source]\n  Agentic RAG adds planning." in transcript
        assert "Retrieved context:" not in transcript
        assert "Prompt preview" in inspector
        assert "Retrieved context:" in inspector
        assert "Agentic RAG adds planning before retrieval." in inspector
        assert app.shell_state.selected_source is not None


@pytest.mark.anyio
async def test_tui_app_shell_ask_missing_vector_index_guidance_is_actionable(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)
    guidance = (
        "Vector index not found.\n"
        "Run `ragent index build` first.\n"
        "Use /mode bm25 or /mode lexical to continue without vectors."
    )

    class FakeWorker:
        name = "shell-ask"
        result = AskPageState(
            question="What is Agentic RAG?",
            retrieval_mode="hybrid",
            error=guidance,
            has_run=True,
        )

    class FakeEvent:
        state = WorkerState.SUCCESS
        worker = FakeWorker()
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    async with app.run_test():
        app.shell_state = replace(app.shell_state, running=True)

        event = FakeEvent()
        app._handle_shell_ask_worker_state(event)  # type: ignore[arg-type]

        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        status = str(app.query_one("#shell-status", Static).renderable)
        inspector = str(app.query_one("#inspector-content", Static).renderable)
        assert "Assistant: [failed]\n  Vector index not found." in transcript
        assert "Vector index not found." not in status
        assert "ragent index build" in inspector
        assert event.stopped is True
        saved = SessionService(workspace.root_path).load_latest_or_create()
        assert saved.turns[-1].assistant_message.text == guidance
        assert saved.turns[-1].run is not None
        assert saved.turns[-1].run.generation_status == "failed"
        assert saved.turns[-1].run.error == guidance


@pytest.mark.anyio
async def test_tui_app_shell_ask_worker_failure_message_is_actionable(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    class FakeWorker:
        name = "shell-ask"
        result = None

    class FakeEvent:
        state = WorkerState.ERROR
        worker = FakeWorker()
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    async with app.run_test():
        app._pending_ask_question = "Why did this fail?"
        app._pending_ask_run = TuiSessionRun(
            retrieval_mode="hybrid",
            retrieval_method="hybrid_rrf",
            limit=5,
            max_context_chars=4000,
            show_prompt=False,
            generation_status="running",
        )
        app.shell_state = replace(app.shell_state, running=True)

        event = FakeEvent()
        app._handle_shell_ask_worker_state(event)  # type: ignore[arg-type]

        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "Try /settings" in transcript
        assert "/docs" in transcript
        assert "/mode bm25" in transcript
        assert event.stopped is True


@pytest.mark.anyio
async def test_tui_app_shell_search_worker_failure_message_is_actionable(
    tmp_path: Path,
) -> None:
    workspace = make_tui_workspace(tmp_path)
    app = RagentForgeApp(workspace.root_path)

    class FakeWorker:
        name = "shell-search"
        result = None

    class FakeEvent:
        state = WorkerState.ERROR
        worker = FakeWorker()
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    async with app.run_test():
        app.shell_state = replace(app.shell_state, running=True)

        event = FakeEvent()
        app._handle_shell_search_worker_state(event)  # type: ignore[arg-type]

        status = str(app.query_one("#shell-status", Static).renderable)
        assert "Try /settings" in status
        assert "/docs" in status
        assert "/mode bm25" in status
        assert event.stopped is True


@pytest.mark.anyio
async def test_tui_app_shell_search_starts_worker_and_keeps_input_enabled(
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
        assert shell_input.disabled is False
        assert app.focused == shell_input
        assert app.shell_state.running is True
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "Running search for: Agentic" not in transcript
        assert "status: running" in str(
            app.query_one("#shell-status", Static).renderable
        )


@pytest.mark.anyio
async def test_tui_app_shell_search_worker_success_opens_source_picker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    opened_sources: list[tuple[TranscriptSource, ...]] = []

    def fake_show_sources_modal() -> None:
        opened_sources.append(app.shell_state.available_sources)

    monkeypatch.setattr(app, "_show_sources_modal", fake_show_sources_modal)

    async with app.run_test():
        app._set_shell_running(True)

        event = FakeEvent()
        app._handle_shell_search_worker_state(event)  # type: ignore[arg-type]

        assert event.stopped is True
        assert app.shell_state.running is False
        assert app.query_one("#shell-input", Input).disabled is False
        transcript = str(app.query_one("#shell-transcript", Static).renderable)
        assert "Search results for: Agentic" not in transcript
        assert "agentic_rag.md" not in transcript
        assert opened_sources
        assert opened_sources[0][0].chunk_id == source.chunk_id


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
