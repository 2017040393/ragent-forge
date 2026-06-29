from textual.widgets import Static


class TraceScreen(Static):
    DEFAULT_CSS = "TraceScreen { padding: 1; }"

    def __init__(self) -> None:
        super().__init__("Trace screen placeholder")
