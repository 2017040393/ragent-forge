from __future__ import annotations

from ragent_forge.app.models import ContextChunk, ContextPack
from ragent_forge.app.services.search_service import SearchResult


def build_context_pack(
    question: str,
    results: list[SearchResult],
    max_context_chars: int = 4000,
) -> ContextPack:
    remaining_chars = max(0, max_context_chars)
    context_chunks: list[ContextChunk] = []
    total_context_chars = 0

    for result in results:
        if remaining_chars <= 0:
            break

        text = result.text[:remaining_chars]
        context_chunks.append(
            ContextChunk(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                source_path=result.source_path,
                start_char=result.start_char,
                end_char=result.end_char,
                score=result.score,
                text=text,
                metadata=result.metadata,
            )
        )
        text_length = len(text)
        total_context_chars += text_length
        remaining_chars -= text_length

    context_pack = ContextPack(
        question=question,
        context_chunks=context_chunks,
        total_context_chars=total_context_chars,
        prompt_preview="",
        metadata={"max_context_chars": max_context_chars},
    )
    return context_pack.model_copy(
        update={"prompt_preview": build_prompt_preview(context_pack)}
    )


def build_prompt_preview(context_pack: ContextPack) -> str:
    return build_generation_prompt(context_pack)


def build_generation_prompt(context_pack: ContextPack) -> str:
    lines = [
        "You are RAGentForge, a local retrieval-augmented assistant.",
        "Use only the retrieved context below to answer the question.",
        "Do not use outside knowledge.",
        "Do not invent details.",
        "Do not invent sources.",
        "Answer clearly and concisely.",
        'If the retrieved context is insufficient, say:',
        '"I cannot determine the answer from the provided context."',
        "",
        "Question:",
        context_pack.question,
        "",
        "Retrieved context:",
    ]

    if not context_pack.context_chunks:
        lines.append("No retrieved context.")
    else:
        for index, chunk in enumerate(context_pack.context_chunks, start=1):
            lines.extend(
                [
                    f"[{index}] Source: {chunk.source_path}",
                    f"Chunk ID: {chunk.chunk_id}",
                    f"Score: {chunk.score:g}",
                    "Content:",
                    chunk.text,
                    "",
                ]
            )
    return "\n".join(lines)
