from textual.widgets import Static


class AskScreen(Static):
    DEFAULT_CSS = "AskScreen { padding: 1; }"

    def __init__(self) -> None:
        super().__init__("Ask screen placeholder")
