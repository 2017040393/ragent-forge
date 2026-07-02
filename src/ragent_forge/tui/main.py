from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)
from textual.worker import Worker, WorkerState

from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.tui.shell_dispatch import ShellReadOnlyHandlers, apply_shell_input
from ragent_forge.tui.shell_models import (
    ShellState,
    TranscriptMessage,
    append_message,
    append_messages,
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
    ChunkRow,
    DocumentsPageModel,
    PageName,
    SearchPageState,
    TracePageModel,
    compact_chunk_label,
    compact_source_label,
    format_ask_response_panel,
    format_ask_source_inspector,
    format_ask_status,
    format_chunk_inspector,
    format_documents_page,
    format_search_result_inspector,
    format_search_status,
    format_settings_page,
    format_trace_inspector,
    format_trace_overview,
    format_trace_steps,
    load_documents_page_model,
    load_settings_page_model,
    load_trace_page_model,
    page_for_key,
    run_tui_ask,
    run_tui_search,
)

ASK_RUNNING_STATUS = "Running ask..."
ASK_FAILED_STATUS = "Ask failed. Check configuration and workspace files."
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

    #navigation {
        width: 22;
        padding: 1;
        border: solid $primary;
    }

    #main-panel {
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

    #search-controls {
        height: auto;
        margin-bottom: 1;
    }

    #ask-controls {
        height: auto;
        margin-bottom: 1;
    }

    #ask-message {
        height: auto;
        margin-bottom: 1;
    }

    #ask-answer-container {
        height: 12;
        margin-bottom: 1;
        padding: 1;
        border: solid $secondary;
    }

    #ask-answer {
        height: auto;
    }

    #ask-sources-table {
        height: 1fr;
    }

    #query-input {
        margin-bottom: 1;
    }

    .nav-button {
        width: 100%;
        margin-bottom: 1;
    }

    .nav-button.active {
        text-style: bold;
    }

    .page {
        height: 1fr;
    }

    .hidden {
        display: none;
    }

    DataTable {
        height: 1fr;
    }

    Static {
        width: 100%;
    }
    """

    BINDINGS = [
        ("h", "switch_page('shell')", "Shell"),
        ("1", "switch_page('shell')", "Shell"),
        ("d", "switch_page('documents')", "Documents"),
        ("2", "switch_page('documents')", "Documents"),
        ("s", "switch_page('search')", "Search"),
        ("3", "switch_page('search')", "Search"),
        ("a", "switch_page('ask')", "Ask"),
        ("4", "switch_page('ask')", "Ask"),
        ("t", "switch_page('trace')", "Trace"),
        ("5", "switch_page('trace')", "Trace"),
        ("g", "switch_page('settings')", "Settings"),
        ("6", "switch_page('settings')", "Settings"),
        ("r", "refresh_page", "Refresh"),
        ("/", "focus_search", "Focus input"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, workspace_path: str | Path = ".ragent") -> None:
        super().__init__()
        self.workspace_path = workspace_path
        self.current_page: PageName = "documents"
        self.shell_state: ShellState = create_initial_shell_state()
        self.documents_model: DocumentsPageModel | None = None
        self.search_state = SearchPageState()
        self.ask_state = AskPageState()
        self.ask_running = False
        self.trace_model: TracePageModel | None = None
        self.selected_chunk: ChunkRow | None = None
        self.selected_search_result: SearchResult | None = None
        self.selected_ask_source: SearchResult | None = None
        self.selected_trace: dict[str, object] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="navigation"):
                yield Label("Navigation")
                yield Button("1 Shell", id="nav-shell", classes="nav-button")
                yield Button("2 Documents", id="nav-documents", classes="nav-button")
                yield Button("3 Search", id="nav-search", classes="nav-button")
                yield Button("4 Ask", id="nav-ask", classes="nav-button")
                yield Button("5 Trace", id="nav-trace", classes="nav-button")
                yield Button("6 Settings", id="nav-settings", classes="nav-button")
                yield Static("Eval", classes="nav-placeholder")
            with Vertical(id="main-panel"):
                with Vertical(id="shell-page", classes="page hidden"):
                    yield Label("Shell")
                    yield Static("", id="shell-status")
                    with ScrollableContainer(id="shell-transcript-container"):
                        yield Static("", id="shell-transcript")
                    yield Input(
                        placeholder="Ask a question or type /help",
                        id="shell-input",
                    )
                with Vertical(id="documents-page", classes="page"):
                    yield Label("Documents")
                    yield Static("", id="documents-summary")
                    yield DataTable(id="documents-table", cursor_type="row")
                with Vertical(id="search-page", classes="page hidden"):
                    yield Label("Search")
                    with Vertical(id="search-controls"):
                        yield Input(
                            placeholder="Search generated chunks",
                            id="query-input",
                        )
                        yield Select(
                            [
                                ("lexical", "lexical"),
                                ("semantic", "semantic"),
                                ("hybrid", "hybrid"),
                            ],
                            value="lexical",
                            allow_blank=False,
                            id="retrieval-mode",
                        )
                        yield Input(value="5", placeholder="Limit", id="limit-input")
                        yield Button("Run Search", id="run-search")
                    yield Static("", id="search-message")
                    yield DataTable(id="search-results-table", cursor_type="row")
                with Vertical(id="ask-page", classes="page hidden"):
                    yield Label("Ask")
                    with Vertical(id="ask-controls"):
                        yield Input(
                            placeholder="Ask about generated chunks",
                            id="ask-question-input",
                        )
                        yield Select(
                            [
                                ("lexical", "lexical"),
                                ("semantic", "semantic"),
                                ("hybrid", "hybrid"),
                            ],
                            value="lexical",
                            allow_blank=False,
                            id="ask-retrieval-mode",
                        )
                        yield Input(
                            value="5",
                            placeholder="Limit",
                            id="ask-limit-input",
                        )
                        yield Input(
                            value="4000",
                            placeholder="Max context chars",
                            id="ask-max-context-input",
                        )
                        yield Select(
                            [("false", "false"), ("true", "true")],
                            value="false",
                            allow_blank=False,
                            id="ask-show-prompt",
                        )
                        yield Button("Run Ask", id="run-ask")
                    yield Static("", id="ask-message")
                    with ScrollableContainer(id="ask-answer-container"):
                        yield Static("", id="ask-answer")
                    yield DataTable(id="ask-sources-table", cursor_type="row")
                with Vertical(id="trace-page", classes="page hidden"):
                    yield Label("Trace")
                    yield Static("", id="trace-summary")
                    yield DataTable(id="trace-table", cursor_type="row")
                    yield Static("", id="trace-steps")
                with Vertical(id="settings-page", classes="page hidden"):
                    yield Label("Settings")
                    yield Static("", id="settings-summary")
            with Vertical(id="inspector"):
                yield Label("Inspector")
                yield Static("", id="inspector-content")
        yield Static(
            "Keys: 1/h Shell | 2/d Documents | 3/s Search | 4/a Ask | "
            "5/t Trace | 6/g Settings | / focus input | "
            "Enter run/select | r refresh | q quit",
            id="help",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._setup_tables()
        self._refresh_all()
        self._show_page("documents")

    def _setup_tables(self) -> None:
        documents_table = self.query_one("#documents-table", DataTable)
        documents_table.add_columns("#", "Source", "Range", "Preview")
        search_table = self.query_one("#search-results-table", DataTable)
        search_table.add_columns("#", "Score", "Source", "Chunk")
        ask_table = self.query_one("#ask-sources-table", DataTable)
        ask_table.add_columns("#", "Score", "Source", "Chunk")
        trace_table = self.query_one("#trace-table", DataTable)
        trace_table.add_columns("#", "Operation", "Status", "Started at")

    def _refresh_all(self) -> None:
        self._load_documents()
        self._load_trace()
        self._render_shell()
        self._render_settings()
        self._render_search()
        self._render_ask()
        self._render_inspector()

    def action_switch_page(self, page: str) -> None:
        resolved_page = page_for_key(page) or page
        if resolved_page in {
            "shell",
            "documents",
            "search",
            "ask",
            "trace",
            "settings",
        }:
            self._show_page(resolved_page)

    def action_refresh_page(self) -> None:
        if self.current_page == "shell":
            self._render_shell()
        elif self.current_page == "documents":
            self._load_documents()
        elif self.current_page == "trace":
            self._load_trace()
        elif self.current_page == "settings":
            self._render_settings()
        elif self.current_page == "search":
            self._render_search()
        elif self.current_page == "ask":
            self._render_ask()
        self._render_inspector()

    def action_focus_search(self) -> None:
        if self.current_page == "shell":
            self.set_focus(self.query_one("#shell-input", Input))
        elif self.current_page == "search":
            self.set_focus(self.query_one("#query-input", Input))
        elif self.current_page == "ask":
            self.set_focus(self.query_one("#ask-question-input", Input))
        else:
            self._show_page("search")
            self.set_focus(self.query_one("#query-input", Input))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        page_by_button = {
            "nav-shell": "shell",
            "nav-documents": "documents",
            "nav-search": "search",
            "nav-ask": "ask",
            "nav-trace": "trace",
            "nav-settings": "settings",
        }
        if button_id in page_by_button:
            self._show_page(page_by_button[button_id])
            return
        if button_id == "run-search":
            self._run_search_from_inputs()
        elif button_id == "run-ask":
            self._run_ask_from_inputs()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "shell-input":
            self._submit_shell_input()
        elif event.input.id in {"query-input", "limit-input"}:
            self._run_search_from_inputs()
        elif event.input.id in {
            "ask-question-input",
            "ask-limit-input",
            "ask-max-context-input",
        }:
            self._run_ask_from_inputs()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "retrieval-mode":
            self.search_state = SearchPageState(
                query=self.search_state.query,
                retrieval_mode=str(event.value),  # type: ignore[arg-type]
                limit=self.search_state.limit,
                results=self.search_state.results,
                error=self.search_state.error,
                selected_result=self.search_state.selected_result,
                has_searched=self.search_state.has_searched,
            )
            self._render_search()
        elif event.select.id == "ask-retrieval-mode":
            mode = str(event.value)
            if mode not in {"lexical", "semantic", "hybrid"}:
                mode = "lexical"
            self.ask_state = replace(self.ask_state, retrieval_mode=mode)
            self._render_ask()
        elif event.select.id == "ask-show-prompt":
            self.ask_state = replace(
                self.ask_state,
                show_prompt=str(event.value) == "true",
            )
            self._render_ask()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table_id = event.data_table.id
        row_index = event.cursor_row
        if table_id == "documents-table":
            self._select_chunk(row_index)
        elif table_id == "search-results-table":
            self._select_search_result(row_index)
        elif table_id == "ask-sources-table":
            self._select_ask_source(row_index)
        elif table_id == "trace-table":
            self._select_trace(row_index)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "search-results-table":
            self._select_search_result(event.cursor_row)
        elif event.data_table.id == "ask-sources-table":
            self._select_ask_source(event.cursor_row)
        elif event.data_table.id == "documents-table":
            self._select_chunk(event.cursor_row)
        elif event.data_table.id == "trace-table":
            self._select_trace(event.cursor_row)

    def _show_page(self, page: PageName) -> None:
        self.current_page = page
        for page_name in ("shell", "documents", "search", "ask", "trace", "settings"):
            widget = self.query_one(f"#{page_name}-page")
            widget.set_class(page_name != page, "hidden")
        if page == "shell":
            self.set_focus(self.query_one("#shell-input", Input))
        elif page == "search":
            self.set_focus(self.query_one("#query-input", Input))
        elif page == "ask":
            self.set_focus(self.query_one("#ask-question-input", Input))
        self._render_navigation()
        self._render_inspector()

    def _render_navigation(self) -> None:
        labels = {
            "shell": "1 Shell",
            "documents": "2 Documents",
            "search": "3 Search",
            "ask": "4 Ask",
            "trace": "5 Trace",
            "settings": "6 Settings",
        }
        for page_name, label in labels.items():
            button = self.query_one(f"#nav-{page_name}", Button)
            is_active = self.current_page == page_name
            button.label = f"> {label}" if is_active else f"  {label}"
            button.set_class(is_active, "active")

    def _load_documents(self) -> None:
        self.documents_model = load_documents_page_model(self.workspace_path)
        self.selected_chunk = self.documents_model.selected_chunk
        summary = format_documents_page(self.documents_model).split(
            "\n\nRecent chunks",
            1,
        )[0]
        self.query_one("#documents-summary", Static).update(summary)

        table = self.query_one("#documents-table", DataTable)
        table.clear()
        for row in self.documents_model.recent_chunks:
            table.add_row(row.index, row.source_label, row.range_text, row.preview)

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

    def _render_search(self) -> None:
        self.selected_search_result = self.search_state.selected_result
        self.query_one("#query-input", Input).value = self.search_state.query
        self.query_one("#limit-input", Input).value = str(self.search_state.limit)
        mode_select = self.query_one("#retrieval-mode", Select)
        if mode_select.value != self.search_state.retrieval_mode:
            mode_select.value = self.search_state.retrieval_mode

        self.query_one("#search-message", Static).update(
            format_search_status(self.search_state)
        )
        table = self.query_one("#search-results-table", DataTable)
        table.clear()
        for index, result in enumerate(self.search_state.results, start=1):
            table.add_row(
                index,
                f"{result.score:.4g}",
                compact_source_label(result.source_path),
                compact_chunk_label(result.chunk_id),
            )

    def _render_ask(self) -> None:
        self.selected_ask_source = self.ask_state.selected_source
        self.query_one("#ask-question-input", Input).value = self.ask_state.question
        self.query_one("#ask-limit-input", Input).value = str(self.ask_state.limit)
        self.query_one("#ask-max-context-input", Input).value = str(
            self.ask_state.max_context_chars
        )
        mode_select = self.query_one("#ask-retrieval-mode", Select)
        if mode_select.value != self.ask_state.retrieval_mode:
            mode_select.value = self.ask_state.retrieval_mode
        show_prompt_select = self.query_one("#ask-show-prompt", Select)
        show_prompt_value = "true" if self.ask_state.show_prompt else "false"
        if show_prompt_select.value != show_prompt_value:
            show_prompt_select.value = show_prompt_value

        self.query_one("#ask-message", Static).update(
            format_ask_status(self.ask_state)
        )
        self.query_one("#ask-answer", Static).update(
            format_ask_response_panel(self.ask_state)
        )
        table = self.query_one("#ask-sources-table", DataTable)
        table.clear()
        for index, result in enumerate(self.ask_state.sources, start=1):
            table.add_row(
                index,
                f"{result.score:.4g}",
                compact_source_label(result.source_path),
                compact_chunk_label(result.chunk_id),
            )

    def _run_ask_from_inputs(self) -> None:
        if self.ask_running:
            return

        question = self.query_one("#ask-question-input", Input).value
        mode_value = self.query_one("#ask-retrieval-mode", Select).value
        mode = (
            str(mode_value)
            if mode_value in {"lexical", "semantic", "hybrid"}
            else "lexical"
        )
        limit_value = self.query_one("#ask-limit-input", Input).value.strip()
        try:
            limit = int(limit_value)
        except ValueError:
            limit = 5
        max_context_value = self.query_one(
            "#ask-max-context-input",
            Input,
        ).value.strip()
        try:
            max_context_chars = int(max_context_value)
        except ValueError:
            max_context_chars = 4000
        show_prompt = self.query_one("#ask-show-prompt", Select).value == "true"
        self.ask_state = replace(
            self.ask_state,
            question=question,
            retrieval_mode=mode,  # type: ignore[arg-type]
            limit=limit,
            max_context_chars=max_context_chars,
            show_prompt=show_prompt,
            status=ASK_RUNNING_STATUS,
            answer=None,
            sources=[],
            selected_source=None,
            generation_status=None,
            generation_provider=None,
            prompt_preview=None,
            error=None,
            has_run=True,
        )
        self.selected_ask_source = None
        self._set_ask_running(True)
        self._render_ask()
        self._render_inspector()
        self.run_worker(
            lambda: self._run_ask_worker(
                question,
                mode,
                limit,
                max_context_chars,
                show_prompt,
            ),
            name="ask",
            group="ask",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _run_ask_worker(
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

    def _set_ask_running(self, running: bool) -> None:
        self.ask_running = running
        run_button = self.query_one("#run-ask", Button)
        run_button.disabled = running
        run_button.label = "Running..." if running else "Run Ask"

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "shell-ask":
            self._handle_shell_ask_worker_state(event)
            return
        if event.worker.name == "shell-search":
            self._handle_shell_search_worker_state(event)
            return
        if event.worker.name != "ask":
            return
        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            if isinstance(result, AskPageState):
                self.ask_state = result
                self.selected_ask_source = result.selected_source
            else:
                self._show_ask_worker_failure()
        elif event.state in {WorkerState.ERROR, WorkerState.CANCELLED}:
            self._show_ask_worker_failure()
        else:
            return

        self._set_ask_running(False)
        self._render_ask()
        self._render_inspector()
        event.stop()

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
                self.shell_state = append_message(
                    self.shell_state,
                    message,
                )
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

    def _show_ask_worker_failure(self) -> None:
        self.ask_state = replace(
            self.ask_state,
            status=ASK_FAILED_STATUS,
            error=ASK_FAILED_STATUS,
            answer=None,
            sources=[],
            selected_source=None,
            generation_status=None,
            generation_provider=None,
            prompt_preview=None,
            has_run=True,
        )
        self.selected_ask_source = None

    def _run_search_from_inputs(self) -> None:
        query = self.query_one("#query-input", Input).value
        mode_value = self.query_one("#retrieval-mode", Select).value
        mode = (
            str(mode_value)
            if mode_value in {"lexical", "semantic", "hybrid"}
            else "lexical"
        )
        limit_value = self.query_one("#limit-input", Input).value.strip()
        try:
            limit = int(limit_value)
        except ValueError:
            limit = 5
        self.search_state = run_tui_search(self.workspace_path, query, mode, limit)
        self._render_search()
        self._render_inspector()

    def _load_trace(self) -> None:
        self.trace_model = load_trace_page_model(self.workspace_path)
        self.selected_trace = self.trace_model.selected_trace
        self.query_one("#trace-summary", Static).update(
            format_trace_overview(self.selected_trace)
        )
        self.query_one("#trace-steps", Static).update(
            format_trace_steps(self.selected_trace)
        )

        table = self.query_one("#trace-table", DataTable)
        table.clear()
        for index, trace in enumerate(self.trace_model.recent_traces, start=1):
            table.add_row(
                index,
                str(trace.get("operation", "")),
                str(trace.get("status", "")),
                str(trace.get("started_at", "")),
            )

    def _render_settings(self) -> None:
        model = load_settings_page_model(self.workspace_path)
        self.query_one("#settings-summary", Static).update(format_settings_page(model))

    def _select_chunk(self, row_index: int) -> None:
        if self.documents_model is None:
            return
        if 0 <= row_index < len(self.documents_model.recent_chunks):
            self.selected_chunk = self.documents_model.recent_chunks[row_index]
            self._render_inspector()

    def _select_search_result(self, row_index: int) -> None:
        if 0 <= row_index < len(self.search_state.results):
            self.selected_search_result = self.search_state.results[row_index]
            self.search_state = SearchPageState(
                query=self.search_state.query,
                retrieval_mode=self.search_state.retrieval_mode,
                limit=self.search_state.limit,
                results=self.search_state.results,
                error=self.search_state.error,
                selected_result=self.selected_search_result,
                has_searched=self.search_state.has_searched,
            )
            self._render_inspector()

    def _select_ask_source(self, row_index: int) -> None:
        if 0 <= row_index < len(self.ask_state.sources):
            self.selected_ask_source = self.ask_state.sources[row_index]
            self.ask_state = replace(
                self.ask_state,
                selected_source=self.selected_ask_source,
            )
            self._render_inspector()

    def _select_trace(self, row_index: int) -> None:
        if self.trace_model is None:
            return
        if 0 <= row_index < len(self.trace_model.recent_traces):
            self.selected_trace = self.trace_model.recent_traces[row_index]
            self.query_one("#trace-summary", Static).update(
                format_trace_overview(self.selected_trace)
            )
            self.query_one("#trace-steps", Static).update(
                format_trace_steps(self.selected_trace)
            )
            self._render_inspector()

    def _render_inspector(self) -> None:
        content = "Inspector\n\nSelect a chunk, search result, trace, or setting."
        if self.current_page == "shell":
            content = format_shell_inspector(self.shell_state)
        elif self.current_page == "documents":
            content = format_chunk_inspector(self.selected_chunk)
        elif self.current_page == "search":
            content = format_search_result_inspector(
                self.selected_search_result,
                self.search_state.retrieval_mode,
            )
        elif self.current_page == "ask":
            content = format_ask_source_inspector(self.selected_ask_source)
        elif self.current_page == "trace":
            content = format_trace_inspector(self.selected_trace)
        elif self.current_page == "settings":
            content = "Settings details\n\nConfiguration is read-only in this TUI."
        self.query_one("#inspector-content", Static).update(content)


def run() -> None:
    RagentForgeApp().run()
