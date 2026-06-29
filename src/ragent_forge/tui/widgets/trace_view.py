from textual.widgets import Static


class TraceView(Static):
    def __init__(self) -> None:
        super().__init__("Trace steps will appear here for inspectable runs.")
