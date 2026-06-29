from ragent_forge.app.models import RagTrace


class TraceService:
    def create_empty_trace(self, query: str) -> RagTrace:
        return RagTrace(query=query, metadata={"status": "stub"})
