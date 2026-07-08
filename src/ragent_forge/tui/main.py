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

from ragent_forge.tui.commands import (
    complete_tui_command_suggestion,
    count_tui_command_suggestions,
    format_tui_command_help,
    format_tui_command_suggestions,
)
from ragent_forge.tui.shell_dispatch import ShellReadOnlyHandlers, apply_shell_input
from ragent_forge.tui.shell_models import (
    ShellState,
    TranscriptMessage,
    TranscriptSource,
    append_message,
    append_messages,
    create_initial_shell_state,
    format_conversation_transcript,
    format_prompt_preview_inspector,
    format_selected_source_ack,
    format_shell_inspector,
    format_shell_status,
    message_from_search_state,
    messages_from_ask_state,
    select_source,
    set_available_sources,
    set_inspector_text,
    set_notice,
    set_running,
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


class SourcePickerModal(ModalScreen[TranscriptSource | None]):
    BINDINGS = [("escape", "close", "Close")]

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
        if self.selected_source is None:
            return
        for index, source in enumerate(self.sources):
            if source == self.selected_source:
                self.query_one("#source-picker-list", ListView).index = index
                return

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(self.sources[event.index])

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


def _source_picker_label(source: TranscriptSource) -> str:
    label = Path(source.source_path).name or source.source_path
    return f"{source.rank}. {label}  score={source.score:.4g}"


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

    #source-picker-dialog, #help-dialog {
        width: 74;
        max-height: 80%;
        margin: 2 4;
        padding: 1 2;
        border: solid #7aa2f7;
        background: #10141f;
        color: #d7deea;
    }

    #source-picker-list {
        height: auto;
        max-height: 18;
    }

    Static {
        width: 100%;
    }
    """

    BINDINGS = []

    def __init__(self, workspace_path: str | Path = ".ragent") -> None:
        super().__init__()
        self.workspace_path = workspace_path
        self.shell_state: ShellState = create_initial_shell_state()
        self.shell_suggestion_index = 0

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
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input()

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
            return
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
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input()

    def _run_shell_ask_from_dispatch(self, question: str) -> None:
        mode = self.shell_state.retrieval_mode
        limit = self.shell_state.limit
        max_context_chars = self.shell_state.max_context_chars
        show_prompt = self.shell_state.show_prompt
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
        self.shell_state = append_message(
            self.shell_state,
            TranscriptMessage(role="error", text=SHELL_ASK_FAILED_STATUS),
        )

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
        messages = messages_from_ask_state(result)
        index = self._last_streaming_assistant_index()
        if index is None:
            self.shell_state = append_messages(self.shell_state, messages)
        else:
            current_messages = list(self.shell_state.messages)
            replacement_messages = list(messages)
            replacement = (
                replacement_messages.pop(0)
                if replacement_messages
                and replacement_messages[0].role == "assistant"
                else None
            )
            if replacement is None:
                del current_messages[index]
            else:
                current_messages[index] = replacement
            self.shell_state = replace(
                self.shell_state,
                messages=tuple(current_messages),
                notice=None,
            )
            if replacement is not None and replacement.sources:
                self.shell_state = set_available_sources(
                    self.shell_state,
                    replacement.sources,
                )
            self.shell_state = append_messages(
                self.shell_state,
                tuple(replacement_messages),
            )

        for message in messages:
            if message.sources:
                self.shell_state = select_source(
                    self.shell_state,
                    message.sources[0],
                )
                break

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

    def _handle_source_picker_result(
        self,
        source: TranscriptSource | None,
    ) -> None:
        if source is None:
            self._focus_shell_input()
            return
        self.shell_state = set_notice(
            select_source(self.shell_state, source),
            format_selected_source_ack(source),
        )
        self._render_shell()
        self._render_inspector()
        self._focus_shell_input()

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

    def _scroll_transcript_to_end(self) -> None:
        container = self.query_one("#shell-transcript-container", ScrollableContainer)
        container.scroll_end(animate=False, force=True, immediate=True)


def run() -> None:
    RagentForgeApp().run()
