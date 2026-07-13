from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from ragent_forge.app.services.session_service import TuiSessionSummary
from ragent_forge.tui.commands import format_tui_task_help
from ragent_forge.tui.shell_models import TranscriptSource
from ragent_forge.tui.view_models import (
    compact_chunk_label,
    compact_source_label,
)


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
                    ListItem(Label(source_picker_label(source)))
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
            yield Static(format_tui_task_help())

    def action_close(self) -> None:
        self.dismiss(None)


class CommandResultModal(ModalScreen[None]):
    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self.modal_title = title
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="command-result-dialog"):
            yield Label(self.modal_title)
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
                    ListItem(Label(session_picker_label(summary)))
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


def source_picker_label(source: TranscriptSource) -> str:
    label = compact_source_label(source.source_path, source.metadata)
    parts = [f"{source.rank}. {label}"]
    retrieval = source_picker_retrieval_label(source)
    if retrieval:
        parts.append(retrieval)
    parts.extend(
        [
            f"score={source.score:.4g}",
            f"chunk={compact_chunk_label(source.chunk_id)}",
        ]
    )
    return "  ".join(parts)


def source_picker_retrieval_label(source: TranscriptSource) -> str:
    method = source.metadata.get("retrieval_method")
    if isinstance(method, str) and method:
        return f"method={method}"
    mode = source.metadata.get("retrieval_mode")
    if isinstance(mode, str) and mode:
        return f"mode={mode}"
    return ""


def session_picker_label(summary: TuiSessionSummary) -> str:
    markers: list[str] = []
    if summary.pinned:
        markers.append("pin")
    if summary.starred:
        markers.append("star")
    prefix = f"[{','.join(markers)}] " if markers else ""
    metrics = [f"turns={summary.turn_count}"]
    if summary.source_count:
        metrics.append(f"sources={summary.source_count}")
    if summary.failed_turn_count:
        metrics.append(f"failed={summary.failed_turn_count}")
    metrics.append(f"updated={summary.updated_at}")
    return f"{prefix}{summary.title}  {'  '.join(metrics)}"
