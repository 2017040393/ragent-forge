from ragent_forge.app.models import (
    AskResult,
    RagTrace,
    RetrievedChunk,
    SourceRef,
    TraceStep,
)


def test_trace_model_creation() -> None:
    source = SourceRef(
        document_id="/knowledge/rag.md",
        chunk_id="/knowledge/rag.md::chunk-0000",
        source_path="/knowledge/rag.md",
    )
    retrieved = RetrievedChunk(
        chunk_id=source.chunk_id,
        text="Agentic RAG adds planning and tool use to retrieval.",
        source=source,
        score=0.75,
    )
    step = TraceStep(
        name="retrieve",
        description="Retrieve candidate chunks",
        inputs={"query": "What is Agentic RAG?"},
        outputs={"chunks": [source.chunk_id]},
    )

    trace = RagTrace(
        query="What is Agentic RAG?",
        steps=[step],
        retrieved_chunks=[retrieved],
        answer="RAG pipeline not implemented yet.",
        latency_ms=0.0,
        metadata={"mode": "stub"},
    )
    assert trace.answer is not None
    result = AskResult(answer=trace.answer, sources=[source], trace=trace)

    assert result.answer == "RAG pipeline not implemented yet."
    assert result.sources == [source]
    assert result.trace.steps[0].name == "retrieve"
    assert result.model_dump()["trace"]["retrieved_chunks"][0]["score"] == 0.75
