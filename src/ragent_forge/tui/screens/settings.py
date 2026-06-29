from textual.widgets import Static


class SettingsScreen(Static):
    DEFAULT_CSS = "SettingsScreen { padding: 1; }"

    def __init__(self) -> None:
        super().__init__("Settings screen placeholder")
