from textual.widgets import Static


class DocumentsScreen(Static):
    DEFAULT_CSS = "DocumentsScreen { padding: 1; }"

    def __init__(self) -> None:
        super().__init__("Documents screen placeholder")
