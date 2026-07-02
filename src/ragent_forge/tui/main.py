from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
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

from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.tui.view_models import (
    ChunkRow,
    DocumentsPageModel,
    PageName,
    SearchPageState,
    TracePageModel,
    compact_chunk_label,
    compact_source_label,
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
    run_tui_search,
)


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

    #search-controls {
        height: auto;
        margin-bottom: 1;
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
        ("d", "switch_page('documents')", "Documents"),
        ("1", "switch_page('documents')", "Documents"),
        ("s", "switch_page('search')", "Search"),
        ("2", "switch_page('search')", "Search"),
        ("t", "switch_page('trace')", "Trace"),
        ("3", "switch_page('trace')", "Trace"),
        ("g", "switch_page('settings')", "Settings"),
        ("4", "switch_page('settings')", "Settings"),
        ("r", "refresh_page", "Refresh"),
        ("/", "focus_search", "Search query"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, workspace_path: str | Path = ".ragent") -> None:
        super().__init__()
        self.workspace_path = workspace_path
        self.current_page: PageName = "documents"
        self.documents_model: DocumentsPageModel | None = None
        self.search_state = SearchPageState()
        self.trace_model: TracePageModel | None = None
        self.selected_chunk: ChunkRow | None = None
        self.selected_search_result: SearchResult | None = None
        self.selected_trace: dict[str, object] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="navigation"):
                yield Label("Navigation")
                yield Button("1 Documents", id="nav-documents", classes="nav-button")
                yield Button("2 Search", id="nav-search", classes="nav-button")
                yield Button("3 Trace", id="nav-trace", classes="nav-button")
                yield Button("4 Settings", id="nav-settings", classes="nav-button")
                yield Static("Ask", classes="nav-placeholder")
                yield Static("Eval", classes="nav-placeholder")
            with Vertical(id="main-panel"):
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
            "Keys: 1/d Documents | 2/s Search | 3/t Trace | 4/g Settings | "
            "/ query | Enter run/select | r refresh | q quit",
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
        trace_table = self.query_one("#trace-table", DataTable)
        trace_table.add_columns("#", "Operation", "Status", "Started at")

    def _refresh_all(self) -> None:
        self._load_documents()
        self._load_trace()
        self._render_settings()
        self._render_search()
        self._render_inspector()

    def action_switch_page(self, page: str) -> None:
        resolved_page = page_for_key(page) or page
        if resolved_page in {"documents", "search", "trace", "settings"}:
            self._show_page(resolved_page)

    def action_refresh_page(self) -> None:
        if self.current_page == "documents":
            self._load_documents()
        elif self.current_page == "trace":
            self._load_trace()
        elif self.current_page == "settings":
            self._render_settings()
        elif self.current_page == "search":
            self._render_search()
        self._render_inspector()

    def action_focus_search(self) -> None:
        if self.current_page == "search":
            self.set_focus(self.query_one("#query-input", Input))
        else:
            self._show_page("search")
            self.set_focus(self.query_one("#query-input", Input))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        page_by_button = {
            "nav-documents": "documents",
            "nav-search": "search",
            "nav-trace": "trace",
            "nav-settings": "settings",
        }
        if button_id in page_by_button:
            self._show_page(page_by_button[button_id])
            return
        if button_id == "run-search":
            self._run_search_from_inputs()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {"query-input", "limit-input"}:
            self._run_search_from_inputs()

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

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table_id = event.data_table.id
        row_index = event.cursor_row
        if table_id == "documents-table":
            self._select_chunk(row_index)
        elif table_id == "search-results-table":
            self._select_search_result(row_index)
        elif table_id == "trace-table":
            self._select_trace(row_index)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "search-results-table":
            self._select_search_result(event.cursor_row)
        elif event.data_table.id == "documents-table":
            self._select_chunk(event.cursor_row)
        elif event.data_table.id == "trace-table":
            self._select_trace(event.cursor_row)

    def _show_page(self, page: PageName) -> None:
        self.current_page = page
        for page_name in ("documents", "search", "trace", "settings"):
            widget = self.query_one(f"#{page_name}-page")
            widget.set_class(page_name != page, "hidden")
        if page == "search":
            self.set_focus(self.query_one("#query-input", Input))
        self._render_navigation()
        self._render_inspector()

    def _render_navigation(self) -> None:
        labels = {
            "documents": "1 Documents",
            "search": "2 Search",
            "trace": "3 Trace",
            "settings": "4 Settings",
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
        if self.current_page == "documents":
            content = format_chunk_inspector(self.selected_chunk)
        elif self.current_page == "search":
            content = format_search_result_inspector(
                self.selected_search_result,
                self.search_state.retrieval_mode,
            )
        elif self.current_page == "trace":
            content = format_trace_inspector(self.selected_trace)
        elif self.current_page == "settings":
            content = "Settings details\n\nConfiguration is read-only in this TUI."
        self.query_one("#inspector-content", Static).update(content)


def run() -> None:
    RagentForgeApp().run()
