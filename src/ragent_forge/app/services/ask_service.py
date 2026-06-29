from ragent_forge.app.models import AskResult, RagTrace


class AskService:
    def ask(self, question: str) -> AskResult:
        trace = RagTrace(
            query=question,
            answer="The inspectable RAG pipeline is not implemented yet.",
            latency_ms=0.0,
            metadata={"status": "stub"},
        )
        return AskResult(answer=trace.answer or "", sources=[], trace=trace)
