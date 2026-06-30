from ragent_forge.app.services.context_service import (
    build_context_pack,
    build_prompt_preview,
)
from ragent_forge.app.services.search_service import SearchResult


def make_search_result(
    chunk_id: str,
    text: str,
    score: float,
    source_path: str = "/knowledge/rag.md",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=source_path,
        source_path=source_path,
        start_char=0,
        end_char=len(text),
        score=score,
        text=text,
    )


def test_build_context_pack_preserves_results_and_builds_prompt_preview() -> None:
    results = [
        make_search_result("/knowledge/rag.md::chunk-0001", "first context", 3.0),
        make_search_result("/knowledge/rag.md::chunk-0002", "second context", 1.0),
    ]

    context_pack = build_context_pack("What is Agentic RAG?", results)

    assert context_pack.question == "What is Agentic RAG?"
    assert context_pack.retrieval_method == "lexical_token_overlap"
    assert [chunk.chunk_id for chunk in context_pack.context_chunks] == [
        "/knowledge/rag.md::chunk-0001",
        "/knowledge/rag.md::chunk-0002",
    ]
    assert context_pack.total_context_chars == len("first context") + len(
        "second context"
    )
    assert context_pack.generation_status == "not_implemented"
    assert "Question:\nWhat is Agentic RAG?" in context_pack.prompt_preview


def test_build_prompt_preview_includes_instruction_sources_chunks_and_content() -> None:
    context_pack = build_context_pack(
        "What is Agentic RAG?",
        [
            make_search_result(
                "/knowledge/rag.md::chunk-0001",
                "Agentic RAG uses planning.",
                3.0,
            )
        ],
    )

    prompt_preview = build_prompt_preview(context_pack)

    assert "You are a local retrieval-augmented assistant." in prompt_preview
    assert "Use only the retrieved context below." in prompt_preview
    assert "Question:\nWhat is Agentic RAG?" in prompt_preview
    assert "[1] Source: /knowledge/rag.md" in prompt_preview
    assert "Chunk ID: /knowledge/rag.md::chunk-0001" in prompt_preview
    assert "Content:\nAgentic RAG uses planning." in prompt_preview
    assert "Generation is not implemented yet." in prompt_preview


def test_build_context_pack_truncates_long_context_deterministically() -> None:
    context_pack = build_context_pack(
        "What is Agentic RAG?",
        [
            make_search_result("/knowledge/rag.md::chunk-0001", "abcdefghij", 3.0),
            make_search_result("/knowledge/rag.md::chunk-0002", "klmnopqrst", 1.0),
        ],
        max_context_chars=12,
    )

    assert context_pack.total_context_chars == 12
    assert [chunk.text for chunk in context_pack.context_chunks] == ["abcdefghij", "kl"]
    assert "Content:\nabcdefghij" in context_pack.prompt_preview
    assert "Content:\nkl" in context_pack.prompt_preview
    assert "klm" not in context_pack.prompt_preview


def test_build_context_pack_handles_empty_results() -> None:
    context_pack = build_context_pack("What is Agentic RAG?", [])

    assert context_pack.context_chunks == []
    assert context_pack.total_context_chars == 0
    assert "Retrieved context:\nNo retrieved context." in context_pack.prompt_preview
    assert "Generation is not implemented yet." in context_pack.prompt_preview
