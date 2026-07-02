from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Footer, Header, Input, Label, Static
from textual.worker import Worker, WorkerState

from ragent_forge.tui.shell_dispatch import ShellReadOnlyHandlers, apply_shell_input
from ragent_forge.tui.shell_models import (
    ShellState,
    TranscriptMessage,
    append_message,
    append_messages,
    clear_transcript,
    create_initial_shell_state,
    format_shell_inspector,
    format_shell_status,
    format_transcript,
    message_from_search_state,
    messages_from_ask_state,
    select_source,
    set_running,
)
from ragent_forge.tui.view_models import (
    AskPageState,
    SearchPageState,
    format_documents_page,
    format_settings_page,
    format_trace_overview,
    format_trace_steps,
    load_documents_page_model,
    load_settings_page_model,
    load_trace_page_model,
    run_tui_ask,
    run_tui_search,
)

SHELL_ASK_FAILED_STATUS = "Ask failed. Check configuration and workspace files."
SHELL_SEARCH_FAILED_STATUS = "Search failed. Check configuration and workspace files."


class RagentForgeApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #workspace {
        height: 1fr;
    }

    #main-shell-panel {
        width: 1fr;
        padding: 1;
        border: solid $accent;
    }

    #inspector {
        width: 34;
        padding: 1;
        border: solid $secondary;
    }

    #help {
        height: 1;
        padding-left: 1;
    }

    #shell-status {
        height: auto;
        margin-bottom: 1;
    }

    #shell-transcript-container {
        height: 1fr;
        border: solid $secondary;
        padding: 1;
        margin-bottom: 1;
    }

    #shell-input {
        height: auto;
    }

    Static {
        width: 100%;
    }
    """

    BINDINGS = [
        ("/", "focus_shell_input", "Focus input"),
        ("ctrl+l", "clear_shell", "Clear"),
        ("r", "refresh_shell", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, workspace_path: str | Path = ".ragent") -> None:
        super().__init__()
        self.workspace_path = workspace_path
        self.shell_state: ShellState = create_initial_shell_state()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="main-shell-panel"):
                yield Label("Shell")
                yield Static("", id="shell-status")
                with ScrollableContainer(id="shell-transcript-container"):
                    yield Static("", id="shell-transcript")
                yield Input(
                    placeholder="Ask a question or type /help",
                    id="shell-input",
                )
            with Vertical(id="inspector"):
                yield Label("Inspector")
                yield Static("", id="inspector-content")
        yield Static(
            "Keys: / focus input | Ctrl+L clear | r refresh | q quit",
            id="help",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._render_shell()
        self._render_inspector()
        self.set_focus(self.query_one("#shell-input", Input))

    def action_focus_shell_input(self) -> None:
        self.set_focus(self.query_one("#shell-input", Input))

    def action_refresh_shell(self) -> None:
        self._render_shell()
        self._render_inspector()

    def action_clear_shell(self) -> None:
        self.shell_state = clear_transcript(self.shell_state)
        self._set_shell_running(False)
        self._render_shell()
        self._render_inspector()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "shell-input":
            self._submit_shell_input()

    def _render_shell(self) -> None:
        self.query_one("#shell-status", Static).update(
            format_shell_status(self.shell_state)
        )
        self.query_one("#shell-transcript", Static).update(
            format_transcript(self.shell_state.messages)
        )

    def _submit_shell_input(self) -> None:
        shell_input = self.query_one("#shell-input", Input)
        text = shell_input.value
        shell_input.value = ""
        if self.shell_state.running:
            return

        result = apply_shell_input(
            self.shell_state,
            text,
            handlers=self._shell_read_only_handlers(),
        )
        self.shell_state = result.state
        if result.action == "quit":
            self.exit()
            return
        if result.action == "search" and result.search_query is not None:
            self._run_shell_search_from_dispatch(result.search_query)
            return
        if result.action == "ask" and result.ask_question is not None:
            self._run_shell_ask_from_dispatch(result.ask_question)
            return
        self._render_shell()
        self._render_inspector()

    def _run_shell_ask_from_dispatch(self, question: str) -> None:
        mode = self.shell_state.retrieval_mode
        limit = self.shell_state.limit
        max_context_chars = self.shell_state.max_context_chars
        show_prompt = self.shell_state.show_prompt
        self.shell_state = append_messages(
            self.shell_state,
            (
                TranscriptMessage(role="user", text=question),
                TranscriptMessage(role="tool", text=f"Running ask for: {question}"),
            ),
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
        self.shell_state = append_message(
            self.shell_state,
            TranscriptMessage(role="tool", text=f"Running search for: {query}"),
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
        self.query_one("#shell-input", Input).disabled = running

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
                messages = messages_from_ask_state(result)
                self.shell_state = append_messages(self.shell_state, messages)
                for message in messages:
                    if message.sources:
                        self.shell_state = select_source(
                            self.shell_state,
                            message.sources[0],
                        )
                        break
            else:
                self._show_shell_ask_worker_failure()
        elif event.state in {WorkerState.ERROR, WorkerState.CANCELLED}:
            self._show_shell_ask_worker_failure()
        else:
            return

        self._set_shell_running(False)
        self._render_shell()
        self._render_inspector()
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
        event.stop()

    def _show_shell_search_worker_failure(self) -> None:
        self.shell_state = append_message(
            self.shell_state,
            TranscriptMessage(role="error", text=SHELL_SEARCH_FAILED_STATUS),
        )

    def _show_shell_ask_worker_failure(self) -> None:
        self.shell_state = append_message(
            self.shell_state,
            TranscriptMessage(role="error", text=SHELL_ASK_FAILED_STATUS),
        )

    def _render_inspector(self) -> None:
        self.query_one("#inspector-content", Static).update(
            format_shell_inspector(self.shell_state)
        )


def run() -> None:
    RagentForgeApp().run()
