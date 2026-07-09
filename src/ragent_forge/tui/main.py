from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Header, Input, Label, ListItem, ListView, Static
from textual.worker import Worker, WorkerState

from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.session_service import (
    SessionService,
    TuiSession,
    TuiSessionExportFormat,
    TuiSessionRun,
    TuiSessionSource,
    TuiSessionSummary,
)
from ragent_forge.app.services.text_generation_client import (
    OpenAIResponsesTextGenerationClient,
)
from ragent_forge.tui.commands import (
    complete_tui_command_suggestion,
    count_tui_command_suggestions,
    format_tui_command_help,
    format_tui_command_suggestions,
)
from ragent_forge.tui.shell_dispatch import (
    ShellDispatchResult,
    ShellReadOnlyHandlers,
    apply_shell_input,
)
from ragent_forge.tui.shell_models import (
    ShellState,
    TranscriptMessage,
    TranscriptSource,
    append_message,
    create_initial_shell_state,
    format_conversation_transcript,
    format_prompt_preview_inspector,
    format_selected_source_ack,
    format_selected_turn_ack,
    format_shell_inspector,
    format_shell_status,
    message_from_search_state,
    messages_from_ask_state,
    replace_state_from_session,
    select_next_turn,
    select_previous_turn,
    select_source,
    select_turn_by_id,
    set_available_sources,
    set_inspector_text,
    set_notice,
    set_running,
    set_session_summaries,
    transcript_sources_from_search_results,
)
from ragent_forge.tui.theme import (
    style_command_suggestions,
    style_inspector,
    style_shell_status,
    style_transcript,
)
from ragent_forge.tui.view_models import (
    AskPageState,
    SearchPageState,
    compact_source_label,
    format_documents_page,
    format_settings_page,
    format_trace_overview,
    format_trace_steps,
    load_documents_page_model,
    load_settings_page_model,
    load_trace_page_model,
    run_tui_ask,
    run_tui_search,
    stream_tui_ask,
)

SHELL_ASK_FAILED_STATUS = "Ask failed. Check configuration and workspace files."
SHELL_SEARCH_FAILED_STATUS = "Search failed. Check configuration and workspace files."
SHELL_RUNNING_DRAFT_QUEUED_STATUS = (
    "Draft queued. Press Enter after the current request finishes to send."
)
SHELL_DRAFT_READY_STATUS = "Draft ready. Press Enter to send."
SHELL_RUNNING_EMPTY_SUBMIT_STATUS = "Request is still running."


class SourcePickerModal(ModalScreen[TranscriptSource | None]):
    BINDINGS = [("escape", "close", "Close"), ("enter", "select", "Open")]

    def __init__(
        self,
        sources: tuple[TranscriptSource, ...],
        selected_source: TranscriptSource | None,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.selected_source = selected_source

    def compose(self) -> ComposeResult:
        with Vertical(id="source-picker-dialog"):
            yield Label("Sources")
            yield Static("Use Up/Down and Enter, or click a source.")
            yield ListView(
                *(
                    ListItem(Label(_source_picker_label(source)))
                    for source in self.sources
                ),
                id="source-picker-list",
            )

    def on_mount(self) -> None:
        source_list = self.query_one("#source-picker-list", ListView)
        if self.selected_source is None:
            source_list.focus()
            return
        for index, source in enumerate(self.sources):
            if source == self.selected_source:
                source_list.index = index
                source_list.focus()
                return
        source_list.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(self.sources[event.index])

    def action_select(self) -> None:
        source_list = self.query_one("#source-picker-list", ListView)
        index = source_list.index
        if index is None or index < 0 or index >= len(self.sources):
            return
        self.dismiss(self.sources[index])

    def action_close(self) -> None:
        self.dismiss(None)


class HelpModal(ModalScreen[None]):
    BINDINGS = [("escape", "close", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Label("Commands")
            yield Static(format_tui_command_help())

    def action_close(self) -> None:
        self.dismiss(None)


class CommandResultModal(ModalScreen[None]):
    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self.title = title
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="command-result-dialog"):
            yield Label(self.title)
            with ScrollableContainer(id="command-result-scroll"):
                yield Static(self.text)

    def action_close(self) -> None:
        self.dismiss(None)


class SessionPickerModal(ModalScreen[str | None]):
    BINDINGS = [("escape", "close", "Close"), ("enter", "select", "Open")]

    def __init__(
        self,
        summaries: tuple[TuiSessionSummary, ...],
        selected_session_id: str | None,
    ) -> None:
        super().__init__()
        self.summaries = summaries
        self.selected_session_id = selected_session_id

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker-dialog"):
            yield Label("Sessions")
            yield Static("Use Up/Down and Enter, or click a session.")
            yield ListView(
                *(
                    ListItem(Label(_session_picker_label(summary)))
                    for summary in self.summaries
                ),
                id="session-picker-list",
            )

    def on_mount(self) -> None:
        session_list = self.query_one("#session-picker-list", ListView)
        if self.selected_session_id is None:
            session_list.focus()
            return
        for index, summary in enumerate(self.summaries):
            if summary.id == self.selected_session_id:
                session_list.index = index
                session_list.focus()
                return
        session_list.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(self.summaries[event.index].id)

    def action_select(self) -> None:
        session_list = self.query_one("#session-picker-list", ListView)
        index = session_list.index
        if index is None or index < 0 or index >= len(self.summaries):
            return
        self.dismiss(self.summaries[index].id)

    def action_close(self) -> None:
        self.dismiss(None)


def _source_picker_label(source: TranscriptSource) -> str:
    label = compact_source_label(source.source_path, source.metadata)
    return f"{source.rank}. {label}  score={source.score:.4g}"


def _session_picker_label(summary: TuiSessionSummary) -> str:
    markers: list[str] = []
    if summary.pinned:
        markers.append("pin")
    if summary.starred:
        markers.append("star")
    prefix = f"[{','.join(markers)}] " if markers else ""
    return (
        f"{prefix}{summary.title}  "
        f"turns={summary.turn_count}  updated={summary.updated_at}"
    )


def _retrieval_method_from_mode(mode: str) -> str:
    if mode == "hybrid":
        return "hybrid_rrf"
    if mode == "semantic":
        return "semantic_cosine_similarity"
    if mode == "bm25":
        return "bm25"
    return "lexical_token_overlap"


def _retrieval_method_from_sources(
    sources: tuple[TranscriptSource, ...],
    fallback_mode: str,
) -> str:
    for source in sources:
        method = source.metadata.get("retrieval_method")
        if isinstance(method, str) and method:
            return method
    return _retrieval_method_from_mode(fallback_mode)


def _session_sources_from_transcript_sources(
    sources: tuple[TranscriptSource, ...],
) -> tuple[TuiSessionSource, ...]:
    return tuple(
        TuiSessionSource(
            rank=source.rank,
            chunk_id=source.chunk_id,
            source_path=source.source_path,
            score=source.score,
            preview=source.preview,
            metadata=dict(source.metadata),
        )
        for source in sources
    )


def _run_metadata_from_session_run(
    run: TuiSessionRun,
    *,
    source_count: int,
) -> dict[str, object]:
    return {
        "operation": "ask",
        "retrieval_mode": run.retrieval_mode,
        "retrieval_method": run.retrieval_method,
        "limit": run.limit,
        "max_context_chars": run.max_context_chars,
        "show_prompt": run.show_prompt,
        "generation_status": run.generation_status,
        "generation_provider": run.generation_provider,
        "error": run.error,
        "source_count": source_count,
    }


def _fallback_title_from_question(question: str) -> str:
    title = " ".join(question.split())
    if len(title) <= 80:
        return title
    return f"{title[:77].rstrip()}..."


class RagentForgeApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: #10141f;
        color: #d7deea;
    }

    #workspace {
        height: 1fr;
    }

    #main-shell-panel {
        width: 1fr;
        padding: 1;
        border: solid #4f8cff;
    }

    #inspector {
        width: 34;
        padding: 1;
        border: solid #7aa2f7;
        color: #d7deea;
    }

    Label {
        color: #8bd5ca;
        text-style: bold;
    }

    #shell-status {
        height: auto;
        margin-bottom: 1;
        color: #d7deea;
    }

    #shell-transcript-container {
        height: 1fr;
        border: solid #394760;
        padding: 1;
        margin-bottom: 1;
    }

    #shell-input {
        height: auto;
        border: tall #4f8cff;
    }

    #shell-suggestions {
        height: auto;
        margin-bottom: 1;
        color: #9aa7bd;
    }

    #source-picker-dialog,
    #help-dialog,
    #session-picker-dialog,
    #command-result-dialog {
        width: 74;
        max-height: 80%;
        margin: 2 4;
        padding: 1 2;
        border: solid #7aa2f7;
        background: #10141f;
        color: #d7deea;
    }

    #source-picker-list, #session-picker-list {
        height: auto;
        max-height: 18;
    }

    #command-result-scroll {
        height: auto;
        max-height: 22;
    }

    Static {
        width: 100%;
    }
    """

    BINDINGS = []

    def __init__(self, workspace_path: str | Path = ".ragent") -> None:
        super().__init__()
        self.workspace_path = workspace_path
        self.session_service = SessionService(workspace_path)
        self.shell_state: ShellState = create_initial_shell_state()
        self.shell_suggestion_index = 0
        self._pending_ask_question: str | None = None
        self._pending_ask_run: TuiSessionRun | None = None
        self._pending_delete_session_id: str | None = None
        self._queued_shell_input: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="main-shell-panel"):
                yield Label("Shell")
                yield Static("", id="shell-status")
                with ScrollableContainer(id="shell-transcript-container"):
                    yield Static("", id="shell-transcript")
                yield Static("", id="shell-suggestions")
                yield Input(
                    placeholder="Ask your knowledge base...  / for commands",
                    id="shell-input",
                )
            with Vertical(id="inspector"):
                yield Label("Inspector")
                yield Static("", id="inspector-content")

    def on_mount(self) -> None:
        self._restore_latest_session()
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input()

    def _handle_session_dispatch(self, result: ShellDispatchResult) -> None:
        if result.action != "delete":
            self._pending_delete_session_id = None
        try:
            if result.action == "new":
                self._create_new_session()
            elif result.action == "sessions":
                self._open_sessions(self.session_service.list_sessions())
            elif result.action == "switch" and result.session_id is not None:
                self._switch_session(result.session_id)
            elif result.action == "rename" and result.title is not None:
                self._rename_current_session(result.title)
            elif result.action == "delete":
                self._delete_current_session()
            elif result.action == "pin":
                self._toggle_current_session_pin()
            elif result.action == "star":
                self._toggle_current_session_star()
            elif result.action == "session-search":
                self._search_sessions(result.session_search_query or "")
            elif result.action == "export" and result.export_format is not None:
                self._export_current_session(result.export_format)
            elif result.action == "branch":
                self._branch_current_session()
            elif result.action == "rerun":
                self._rerun_selected_turn()
            elif result.action == "continue-sources":
                self._continue_from_selected_sources()
            elif result.action == "title":
                self._title_current_session(result.title)
            elif result.action == "turn" and result.turn_selector is not None:
                self._select_turn(result.turn_selector)
        except (OSError, ValueError) as exc:
            self.shell_state = set_notice(self.shell_state, str(exc))

        self._render_shell()
        self._render_inspector()
        self._focus_shell_input()

    def _create_new_session(self) -> None:
        session = self.session_service.create_session()
        self._load_session_into_shell(session)
        self.shell_state = set_notice(self.shell_state, "Started a new session.")

    def _open_sessions(self, summaries: list[TuiSessionSummary]) -> None:
        self.shell_state = set_session_summaries(self.shell_state, summaries)
        self._render_shell()
        self._render_inspector()
        self._show_sessions_modal()

    def _switch_session(self, session_id: str) -> None:
        session = self.session_service.load_session(session_id)
        self.session_service.set_latest(session.id)
        self._load_session_into_shell(session)
        self.shell_state = set_notice(
            self.shell_state,
            f"Switched session: {session.title}",
        )

    def _rename_current_session(self, title: str) -> None:
        session_id = self._current_session_id()
        session = self.session_service.rename_session(session_id, title)
        self._load_session_into_shell(session)
        self.shell_state = set_notice(
            self.shell_state,
            f"Renamed session: {session.title}",
        )

    def _delete_current_session(self) -> None:
        session_id = self._current_session_id()
        if self._pending_delete_session_id != session_id:
            session = self.session_service.load_session(session_id)
            self._pending_delete_session_id = session_id
            self.shell_state = set_notice(
                self.shell_state,
                f"Type /delete again to delete session: {session.title}",
            )
            return

        self._pending_delete_session_id = None
        self.session_service.delete_session(session_id)
        session = self.session_service.load_latest_or_create()
        self._load_session_into_shell(session)
        self.shell_state = set_notice(self.shell_state, "Deleted current session.")

    def _toggle_current_session_pin(self) -> None:
        session_id = self._current_session_id()
        session = self.session_service.load_session(session_id)
        updated = self.session_service.set_pinned(session_id, not session.pinned)
        self._load_session_into_shell(updated)
        status = "pinned" if updated.pinned else "unpinned"
        self.shell_state = set_notice(self.shell_state, f"Session {status}.")

    def _toggle_current_session_star(self) -> None:
        session_id = self._current_session_id()
        session = self.session_service.load_session(session_id)
        updated = self.session_service.set_starred(session_id, not session.starred)
        self._load_session_into_shell(updated)
        status = "starred" if updated.starred else "unstarred"
        self.shell_state = set_notice(self.shell_state, f"Session {status}.")

    def _search_sessions(self, query: str) -> None:
        summaries = self.session_service.search_sessions(query)
        self.shell_state = set_notice(
            set_session_summaries(self.shell_state, summaries),
            f"Found {len(summaries)} session(s).",
        )
        self._show_sessions_modal()

    def _export_current_session(self, export_format: TuiSessionExportFormat) -> None:
        path = self.session_service.export_session(
            self._current_session_id(),
            export_format,
        )
        self.shell_state = set_notice(self.shell_state, f"Exported session: {path}")

    def _branch_current_session(self) -> None:
        branch = self.session_service.branch_session(
            self._current_session_id(),
            self.shell_state.selected_turn_id,
        )
        self._load_session_into_shell(branch)
        self.shell_state = set_notice(
            self.shell_state,
            f"Branched session: {branch.title}",
        )

    def _rerun_selected_turn(self) -> None:
        question = self._selected_turn_question()
        if question is None:
            self.shell_state = set_notice(
                self.shell_state,
                "No answer turn selected.",
            )
            return
        self._run_shell_ask_from_dispatch(question)

    def _continue_from_selected_sources(self) -> None:
        if not self.shell_state.available_sources:
            self.shell_state = set_notice(
                self.shell_state,
                "No selected answer sources to continue from.",
            )
            return
        shell_input = self.query_one("#shell-input", Input)
        shell_input.value = "Using the selected sources, "
        shell_input.cursor_position = len(shell_input.value)
        self.shell_state = set_notice(
            self.shell_state,
            "Drafting follow-up from selected sources.",
        )

    def _title_current_session(self, title: str | None) -> None:
        if title is None:
            current = self.shell_state.current_session_title or "Untitled session"
            self.shell_state = set_notice(self.shell_state, f"Session title: {current}")
            return
        if title == "auto":
            generated = self._generate_session_title()
            if generated is None:
                self.shell_state = set_notice(
                    self.shell_state,
                    "Unable to generate a title with current configuration.",
                )
                return
            self._rename_current_session(generated)
            return
        self._rename_current_session(title)

    def _select_turn(self, selector: str) -> None:
        normalized = selector.strip().lower()
        if normalized == "next":
            updated = select_next_turn(self.shell_state)
            self.shell_state = set_notice(
                updated,
                format_selected_turn_ack(updated),
            )
            return
        if normalized == "prev":
            updated = select_previous_turn(self.shell_state)
            self.shell_state = set_notice(
                updated,
                format_selected_turn_ack(updated),
            )
            return
        turn_id = self._turn_id_from_selector(selector)
        self.shell_state = set_notice(
            select_turn_by_id(self.shell_state, turn_id),
            f"selected answer turn: {turn_id}",
        )

    def _turn_id_from_selector(self, selector: str) -> str:
        normalized = selector.strip().lower()
        turn_ids = self._assistant_turn_ids()
        if not turn_ids:
            raise ValueError("No assistant answers available.")
        if normalized == "first":
            return turn_ids[0]
        if normalized == "last":
            return turn_ids[-1]
        if normalized.isdigit():
            index = int(normalized)
            if index <= 0 or index > len(turn_ids):
                raise ValueError(
                    f"Turn number out of range. Available: 1-{len(turn_ids)}."
                )
            return turn_ids[index - 1]
        return selector.strip()

    def _assistant_turn_ids(self) -> list[str]:
        turn_ids: list[str] = []
        for message in self.shell_state.messages:
            if (
                message.role == "assistant"
                and message.turn_id is not None
                and message.turn_id not in turn_ids
            ):
                turn_ids.append(message.turn_id)
        return turn_ids

    def _selected_turn_question(self) -> str | None:
        selected_turn_id = self.shell_state.selected_turn_id
        if selected_turn_id is None:
            return None
        for message in self.shell_state.messages:
            if message.role == "user" and message.turn_id == selected_turn_id:
                return message.text
        return None

    def _current_session_id(self) -> str:
        if self.shell_state.current_session_id is not None:
            return self.shell_state.current_session_id
        session = self.session_service.load_latest_or_create()
        self._load_session_into_shell(session)
        return session.id

    def _restore_latest_session(self) -> None:
        session = self.session_service.load_latest_or_create()
        self._load_session_into_shell(session)

    def _load_session_into_shell(self, session: TuiSession) -> None:
        self._pending_delete_session_id = None
        self.shell_state = replace_state_from_session(
            self.shell_state,
            session,
            self.session_service.list_sessions(),
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "shell-input":
            self._submit_shell_input()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "shell-input":
            self.shell_suggestion_index = 0
            self._render_shell_suggestions(event.value)

    def on_key(self, event: events.Key) -> None:
        if self.shell_state.running:
            return

        shell_input = self.query_one("#shell-input", Input)
        if not shell_input.has_focus:
            return

        if event.key == "down" and self._move_shell_suggestion(1):
            event.prevent_default()
            event.stop()
            return
        if event.key == "up" and self._move_shell_suggestion(-1):
            event.prevent_default()
            event.stop()
            return
        if event.key in {"tab", "enter"} and self._complete_shell_suggestion():
            event.prevent_default()
            event.stop()

    def _render_shell(self) -> None:
        self.query_one("#shell-status", Static).update(
            style_shell_status(format_shell_status(self.shell_state))
        )
        self.query_one("#shell-transcript", Static).update(
            style_transcript(format_conversation_transcript(self.shell_state.messages))
        )
        self._scroll_transcript_to_end()

    def _submit_shell_input(self) -> None:
        shell_input = self.query_one("#shell-input", Input)
        text = shell_input.value
        if self.shell_state.running:
            if text.strip():
                self._queued_shell_input = text
                notice = SHELL_RUNNING_DRAFT_QUEUED_STATUS
            else:
                notice = SHELL_RUNNING_EMPTY_SUBMIT_STATUS
            self.shell_state = set_notice(
                self.shell_state,
                notice,
            )
            self._render_shell()
            self._render_inspector()
            self._focus_shell_input()
            return
        self._queued_shell_input = None
        shell_input.value = ""
        self.shell_suggestion_index = 0
        self._render_shell_suggestions("")

        result = apply_shell_input(
            self.shell_state,
            text,
            handlers=self._shell_read_only_handlers(),
        )
        self.shell_state = result.state
        if result.action == "quit":
            self.exit()
            return
        if result.action in {
            "new",
            "sessions",
            "switch",
            "rename",
            "delete",
            "pin",
            "star",
            "session-search",
            "export",
            "branch",
            "rerun",
            "continue-sources",
            "title",
            "turn",
        }:
            self._handle_session_dispatch(result)
            return
        if result.action == "search" and result.search_query is not None:
            self._run_shell_search_from_dispatch(result.search_query)
            return
        if result.action == "ask" and result.ask_question is not None:
            self._run_shell_ask_from_dispatch(result.ask_question)
            return
        if result.action == "sources":
            self._render_shell()
            self._render_inspector()
            self._show_sources_modal()
            self._focus_shell_input()
            return
        if result.action == "help":
            self._render_shell()
            self._render_inspector()
            self._show_help_modal()
            self._focus_shell_input()
            return
        if result.action == "command-result":
            self._render_shell()
            self._render_inspector()
            if result.modal_title is not None and result.modal_text is not None:
                self._show_command_result_modal(
                    result.modal_title,
                    result.modal_text,
                )
            self._focus_shell_input()
            return
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input()

    def _run_shell_ask_from_dispatch(self, question: str) -> None:
        mode = self.shell_state.retrieval_mode
        limit = self.shell_state.limit
        max_context_chars = self.shell_state.max_context_chars
        show_prompt = self.shell_state.show_prompt
        self._pending_ask_question = question
        self._pending_ask_run = TuiSessionRun(
            retrieval_mode=mode,
            retrieval_method=_retrieval_method_from_mode(mode),
            limit=limit,
            max_context_chars=max_context_chars,
            show_prompt=show_prompt,
            generation_status="running",
        )
        self.shell_state = set_notice(
            append_message(
                append_message(
                    self.shell_state,
                    TranscriptMessage(role="user", text=question),
                ),
                TranscriptMessage(
                    role="assistant",
                    text="",
                    metadata={"operation": "ask", "streaming": True},
                ),
            ),
            f"Running ask for: {question}",
        )
        self._set_shell_running(True)
        self._render_shell()
        self._render_inspector()
        self.run_worker(
            lambda: self._run_shell_ask_worker(
                question,
                mode,
                limit,
                max_context_chars,
                show_prompt,
            ),
            name="shell-ask",
            group="shell",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _run_shell_ask_worker(
        self,
        question: str,
        mode: str,
        limit: int,
        max_context_chars: int,
        show_prompt: bool,
    ) -> AskPageState:
        final_state: AskPageState | None = None
        for event in stream_tui_ask(
            self.workspace_path,
            question,
            mode,
            limit,
            max_context_chars,
            show_prompt,
        ):
            if event.type == "delta":
                if event.text:
                    self._call_from_worker_thread(
                        self._apply_shell_ask_delta,
                        event.text,
                    )
                continue
            if event.type == "done":
                final_state = event.state

        if final_state is not None:
            return final_state
        return run_tui_ask(
            self.workspace_path,
            question,
            mode,
            limit,
            max_context_chars,
            show_prompt,
        )

    def _run_shell_search_from_dispatch(self, query: str) -> None:
        mode = self.shell_state.retrieval_mode
        limit = self.shell_state.limit
        self.shell_state = set_notice(
            self.shell_state,
            f"Running search for: {query}",
        )
        self._set_shell_running(True)
        self._render_shell()
        self._render_inspector()
        self.run_worker(
            lambda: self._run_shell_search_worker(query, mode, limit),
            name="shell-search",
            group="shell",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _run_shell_search_worker(
        self,
        query: str,
        mode: str,
        limit: int,
    ) -> SearchPageState:
        return run_tui_search(self.workspace_path, query, mode, limit)

    def _set_shell_running(self, running: bool) -> None:
        self.shell_state = set_running(self.shell_state, running)
        if not running and self._queued_shell_input:
            self.shell_state = set_notice(self.shell_state, SHELL_DRAFT_READY_STATUS)
        self.query_one("#shell-input", Input).disabled = False

    def _shell_read_only_handlers(self) -> ShellReadOnlyHandlers:
        return ShellReadOnlyHandlers(
            docs=self._shell_docs_summary,
            trace=self._shell_trace_summary,
            settings=self._shell_settings_summary,
        )

    def _shell_docs_summary(self) -> str:
        return format_documents_page(load_documents_page_model(self.workspace_path))

    def _shell_trace_summary(self) -> str:
        model = load_trace_page_model(self.workspace_path)
        return "\n\n".join(
            [
                format_trace_overview(model.selected_trace),
                format_trace_steps(model.selected_trace),
            ]
        )

    def _shell_settings_summary(self) -> str:
        return format_settings_page(load_settings_page_model(self.workspace_path))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "shell-ask":
            self._handle_shell_ask_worker_state(event)
            return
        if event.worker.name == "shell-search":
            self._handle_shell_search_worker_state(event)

    def _handle_shell_ask_worker_state(
        self,
        event: Worker.StateChanged,
    ) -> None:
        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            if isinstance(result, AskPageState):
                self._complete_shell_ask_result(result)
                if result.prompt_preview:
                    self.shell_state = set_inspector_text(
                        self.shell_state,
                        format_prompt_preview_inspector(result.prompt_preview),
                    )
            else:
                self._show_shell_ask_worker_failure()
        elif event.state in {WorkerState.ERROR, WorkerState.CANCELLED}:
            self._show_shell_ask_worker_failure()
        else:
            return

        self._set_shell_running(False)
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input()
        event.stop()

    def _handle_shell_search_worker_state(
        self,
        event: Worker.StateChanged,
    ) -> None:
        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            if isinstance(result, SearchPageState):
                message = message_from_search_state(result)
                self.shell_state = append_message(self.shell_state, message)
                if message.sources:
                    self.shell_state = select_source(
                        self.shell_state,
                        message.sources[0],
                    )
            else:
                self._show_shell_search_worker_failure()
        elif event.state in {WorkerState.ERROR, WorkerState.CANCELLED}:
            self._show_shell_search_worker_failure()
        else:
            return

        self._set_shell_running(False)
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input()
        event.stop()

    def _show_shell_search_worker_failure(self) -> None:
        self.shell_state = append_message(
            self.shell_state,
            TranscriptMessage(role="error", text=SHELL_SEARCH_FAILED_STATUS),
        )

    def _show_shell_ask_worker_failure(self) -> None:
        self._discard_streaming_assistant()
        question = self._pending_ask_question or ""
        run = self._pending_ask_run or TuiSessionRun(
            retrieval_mode=self.shell_state.retrieval_mode,
            retrieval_method=_retrieval_method_from_mode(
                self.shell_state.retrieval_mode,
            ),
            limit=self.shell_state.limit,
            max_context_chars=self.shell_state.max_context_chars,
            show_prompt=self.shell_state.show_prompt,
            generation_status="failed",
            error=SHELL_ASK_FAILED_STATUS,
        )
        failed_run = replace(
            run,
            generation_status="failed",
            error=SHELL_ASK_FAILED_STATUS,
        )
        message = TranscriptMessage(
            role="assistant",
            text=SHELL_ASK_FAILED_STATUS,
            metadata=_run_metadata_from_session_run(failed_run, source_count=0),
        )
        if question:
            session, _turn = self.session_service.append_turn(
                self._current_session_id(),
                question=question,
                assistant_text=SHELL_ASK_FAILED_STATUS,
                sources=(),
                run=failed_run,
            )
            self._load_session_into_shell(session)
        else:
            self.shell_state = append_message(self.shell_state, message)
        self._pending_ask_question = None
        self._pending_ask_run = None

    def _call_from_worker_thread(
        self,
        callback: Callable[..., None],
        *args: object,
    ) -> None:
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            callback(*args)

    def _apply_shell_ask_delta(self, delta: str) -> None:
        messages = list(self.shell_state.messages)
        index = self._last_streaming_assistant_index()
        if index is None:
            messages.append(
                TranscriptMessage(
                    role="assistant",
                    text=delta,
                    metadata={"operation": "ask", "streaming": True},
                )
            )
        else:
            current = messages[index]
            messages[index] = replace(current, text=f"{current.text}{delta}")
        self.shell_state = replace(
            self.shell_state,
            messages=tuple(messages),
            notice="Receiving answer...",
        )
        self._render_shell()

    def _complete_shell_ask_result(self, result: AskPageState) -> None:
        message = self._assistant_message_from_ask_result(result)
        question = result.question or self._pending_ask_question or ""
        if question:
            session, _turn = self.session_service.append_turn(
                self._current_session_id(),
                question=question,
                assistant_text=message.text,
                sources=_session_sources_from_transcript_sources(message.sources),
                run=self._session_run_from_ask_result(result, message),
            )
            self._load_session_into_shell(session)
        else:
            index = self._last_streaming_assistant_index()
            if index is None:
                self.shell_state = append_message(self.shell_state, message)
            else:
                current_messages = list(self.shell_state.messages)
                current_messages[index] = message
                self.shell_state = replace(
                    self.shell_state,
                    messages=tuple(current_messages),
                    notice=None,
                )
                if message.sources:
                    self.shell_state = set_available_sources(
                        self.shell_state,
                        message.sources,
                    )

        if message.sources:
            self.shell_state = select_source(self.shell_state, message.sources[0])
        self._pending_ask_question = None
        self._pending_ask_run = None

    def _assistant_message_from_ask_result(
        self,
        result: AskPageState,
    ) -> TranscriptMessage:
        if result.error:
            sources = transcript_sources_from_search_results(result.sources)
            metadata = {
                "operation": "ask",
                "retrieval_mode": result.retrieval_mode,
                "retrieval_method": _retrieval_method_from_sources(
                    sources,
                    result.retrieval_mode,
                ),
                "limit": result.limit,
                "max_context_chars": result.max_context_chars,
                "show_prompt": result.show_prompt,
                "generation_status": "failed",
                "generation_provider": result.generation_provider,
                "error": result.error,
                "source_count": len(sources),
            }
            return TranscriptMessage(
                role="assistant",
                text=result.error,
                metadata=metadata,
                sources=sources,
            )

        messages = messages_from_ask_state(result)
        for message in messages:
            if message.role == "assistant":
                metadata = dict(message.metadata)
                metadata.setdefault(
                    "retrieval_method",
                    _retrieval_method_from_sources(
                        message.sources,
                        result.retrieval_mode,
                    ),
                )
                metadata.setdefault("limit", result.limit)
                metadata.setdefault("max_context_chars", result.max_context_chars)
                metadata.setdefault("show_prompt", result.show_prompt)
                return replace(message, metadata=metadata)

        sources = transcript_sources_from_search_results(result.sources)
        text = result.status or "Ask completed."
        metadata = {
            "operation": "ask",
            "retrieval_mode": result.retrieval_mode,
            "retrieval_method": _retrieval_method_from_sources(
                sources,
                result.retrieval_mode,
            ),
            "limit": result.limit,
            "max_context_chars": result.max_context_chars,
            "show_prompt": result.show_prompt,
            "generation_status": result.generation_status,
            "generation_provider": result.generation_provider,
            "source_count": len(sources),
        }
        return TranscriptMessage(
            role="assistant",
            text=text,
            metadata=metadata,
            sources=sources,
        )

    def _session_run_from_ask_result(
        self,
        result: AskPageState,
        message: TranscriptMessage,
    ) -> TuiSessionRun:
        metadata = message.metadata
        return TuiSessionRun(
            retrieval_mode=result.retrieval_mode,
            retrieval_method=str(
                metadata.get(
                    "retrieval_method",
                    _retrieval_method_from_sources(
                        message.sources,
                        result.retrieval_mode,
                    ),
                )
            ),
            limit=result.limit,
            max_context_chars=result.max_context_chars,
            show_prompt=result.show_prompt,
            generation_status=str(metadata.get("generation_status") or ""),
            generation_provider=(
                str(metadata["generation_provider"])
                if metadata.get("generation_provider") is not None
                else None
            ),
            error=(
                str(metadata["error"])
                if metadata.get("error") is not None
                else result.error
            ),
            prompt_preview=result.prompt_preview,
        )

    def _discard_streaming_assistant(self) -> None:
        index = self._last_streaming_assistant_index()
        if index is None:
            return
        messages = list(self.shell_state.messages)
        del messages[index]
        self.shell_state = replace(self.shell_state, messages=tuple(messages))

    def _last_streaming_assistant_index(self) -> int | None:
        for index in range(len(self.shell_state.messages) - 1, -1, -1):
            message = self.shell_state.messages[index]
            if (
                message.role == "assistant"
                and message.metadata.get("streaming") is True
            ):
                return index
        return None

    def _show_sources_modal(self) -> None:
        self.push_screen(
            SourcePickerModal(
                self.shell_state.available_sources,
                self.shell_state.selected_source,
            ),
            self._handle_source_picker_result,
        )

    def _show_help_modal(self) -> None:
        self.push_screen(HelpModal())

    def _show_command_result_modal(self, title: str, text: str) -> None:
        self.push_screen(CommandResultModal(title, text))

    def _show_sessions_modal(self) -> None:
        self.push_screen(
            SessionPickerModal(
                self.shell_state.session_summaries,
                self.shell_state.current_session_id,
            ),
            self._handle_session_picker_result,
        )

    def _handle_session_picker_result(self, session_id: str | None) -> None:
        if session_id is None:
            self._focus_shell_input_after_refresh()
            return
        try:
            self._switch_session(session_id)
        except (OSError, ValueError) as exc:
            self.shell_state = set_notice(self.shell_state, str(exc))
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input_after_refresh()

    def _handle_source_picker_result(
        self,
        source: TranscriptSource | None,
    ) -> None:
        if source is None:
            self._focus_shell_input_after_refresh()
            return
        self.shell_state = set_notice(
            select_source(self.shell_state, source),
            format_selected_source_ack(source),
        )
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input_after_refresh()

    def _generate_session_title(self) -> str | None:
        question = self._first_session_question()
        if question is None:
            return None
        try:
            config = ConfigService(self.session_service.workspace).load()
            if config.generation.provider != "openai_responses":
                return _fallback_title_from_question(question)
            client = OpenAIResponsesTextGenerationClient.from_config(config)
            raw_title = client.generate_text(
                (
                    "Create a concise chat title under 8 words. "
                    "Return only the title.\n\n"
                    f"First question: {question}"
                ),
                system_prompt="You write short, plain chat titles.",
            )
        except (OSError, RuntimeError, ValueError):
            return _fallback_title_from_question(question)
        title = " ".join(raw_title.split()).strip('"')
        return title[:80] or _fallback_title_from_question(question)

    def _first_session_question(self) -> str | None:
        for message in self.shell_state.messages:
            if message.role == "user" and message.text.strip():
                return message.text.strip()
        return None

    def _render_inspector(self) -> None:
        self.query_one("#inspector-content", Static).update(
            style_inspector(format_shell_inspector(self.shell_state))
        )

    def _render_shell_suggestions(self, text: str | None = None) -> None:
        if text is None:
            text = self.query_one("#shell-input", Input).value
        suggestion_count = count_tui_command_suggestions(text)
        if suggestion_count:
            self.shell_suggestion_index %= suggestion_count
            selected_index: int | None = self.shell_suggestion_index
        else:
            self.shell_suggestion_index = 0
            selected_index = None
        suggestions = format_tui_command_suggestions(
            text,
            selected_index=selected_index,
        )
        renderable = "" if not suggestions else style_command_suggestions(suggestions)
        self.query_one("#shell-suggestions", Static).update(renderable)

    def _move_shell_suggestion(self, delta: int) -> bool:
        shell_input = self.query_one("#shell-input", Input)
        suggestion_count = count_tui_command_suggestions(shell_input.value)
        if not suggestion_count:
            return False
        self.shell_suggestion_index = (
            self.shell_suggestion_index + delta
        ) % suggestion_count
        self._render_shell_suggestions(shell_input.value)
        return True

    def _complete_shell_suggestion(self) -> bool:
        shell_input = self.query_one("#shell-input", Input)
        completion = complete_tui_command_suggestion(
            shell_input.value,
            selected_index=self.shell_suggestion_index,
        )
        if completion is None:
            return False

        shell_input.value = completion
        shell_input.cursor_position = len(completion)
        self.shell_suggestion_index = 0
        self._render_shell_suggestions("")
        self._focus_shell_input()
        return True

    def _focus_shell_input(self) -> None:
        shell_input = self.query_one("#shell-input", Input)
        if not shell_input.disabled:
            self.set_focus(shell_input)

    def _focus_shell_input_after_refresh(self) -> None:
        self._focus_shell_input()
        self.call_after_refresh(self._focus_shell_input)

    def _scroll_transcript_to_end(self) -> None:
        container = self.query_one("#shell-transcript-container", ScrollableContainer)
        container.scroll_end(animate=False, force=True, immediate=True)


def run() -> None:
    RagentForgeApp().run()
