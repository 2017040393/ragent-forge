from textual.widgets import Static


class AnswerPanel(Static):
    def __init__(self) -> None:
        super().__init__("Answers will appear here once the RAG pipeline exists.")
