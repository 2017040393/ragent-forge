from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Label, Static

from ragent_forge.tui.screens.documents import DocumentsScreen


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
        padding: 1 2;
        border: solid $primary;
    }

    #main-panel {
        width: 1fr;
        padding: 1 2;
        border: solid $accent;
    }

    #inspector {
        width: 28;
        padding: 1 2;
        border: solid $secondary;
    }

    .nav-item {
        height: 1;
        margin-bottom: 1;
    }

    #status {
        height: 1;
        padding-left: 1;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="navigation"):
                yield Label("Navigation")
                yield Static("Documents", classes="nav-item")
                yield Static("Ask", classes="nav-item")
                yield Static("Trace", classes="nav-item")
                yield Static("Settings", classes="nav-item")
            with Vertical(id="main-panel"):
                yield Label("Documents")
                yield DocumentsScreen()
            with Vertical(id="inspector"):
                yield Label("Inspector")
                yield Static("Sources")
                yield Static("Trace")
                yield Static("Metrics")
        yield Static("Status: local | TUI-first | inspectable RAG", id="status")
        yield Footer()


def run() -> None:
    RagentForgeApp().run()
