from ragent_forge.app.services.context_service import (
    build_context_pack,
    build_generation_prompt,
)
from ragent_forge.app.services.search_service import SearchResult


def make_context_pack():
    return build_context_pack(
        "What is Agentic RAG?",
        [
            SearchResult(
                chunk_id="/knowledge/rag.md::chunk-0001",
                document_id="/knowledge/rag.md",
                source_path="/knowledge/rag.md",
                start_char=0,
                end_char=27,
                score=2.0,
                text="Agentic RAG uses planning.",
            )
        ],
    )


def test_build_generation_prompt_includes_retrieved_context_and_instructions() -> None:
    context_pack = make_context_pack()

    prompt = build_generation_prompt(context_pack)

    assert "Use only the retrieved context below" in prompt
    assert "Do not use outside knowledge." in prompt
    assert "I cannot determine the answer from the provided context." in prompt
    assert "Question:\nWhat is Agentic RAG?" in prompt
    assert "Source: /knowledge/rag.md" in prompt
    assert "Chunk ID: /knowledge/rag.md::chunk-0001" in prompt
    assert "Agentic RAG uses planning." in prompt
    assert "Generation is not implemented yet." not in prompt


def test_build_generation_prompt_handles_empty_context() -> None:
    context_pack = build_context_pack("What is Agentic RAG?", [])

    prompt = build_generation_prompt(context_pack)

    assert "Question:\nWhat is Agentic RAG?" in prompt
    assert "Retrieved context:\nNo retrieved context." in prompt
