from __future__ import annotations

from dataclasses import replace

from ragent_forge.app.services.session_service import (
    TuiSessionRun,
    TuiSessionSource,
)
from ragent_forge.tui.shell_models import (
    TranscriptMessage,
    TranscriptSource,
    messages_from_ask_state,
    transcript_sources_from_search_results,
)
from ragent_forge.tui.view_models import AskPageState


def retrieval_method_from_mode(mode: str) -> str:
    if mode == "bm25":
        return "bm25"
    if mode == "semantic":
        return "semantic_cosine_similarity"
    if mode == "hybrid":
        return "hybrid_rrf"
    return "lexical_token_overlap"


def retrieval_method_from_sources(
    sources: tuple[TranscriptSource, ...],
    fallback_mode: str,
) -> str:
    for source in sources:
        method = source.metadata.get("retrieval_method")
        if isinstance(method, str) and method:
            return method
    return retrieval_method_from_mode(fallback_mode)


def session_sources_from_transcript_sources(
    sources: tuple[TranscriptSource, ...],
) -> tuple[TuiSessionSource, ...]:
    return tuple(
        TuiSessionSource(
            rank=source.rank,
            chunk_id=source.chunk_id,
            source_path=source.source_path,
            score=source.score,
            preview=source.preview,
            metadata=dict(source.metadata),
        )
        for source in sources
    )


def run_metadata_from_session_run(
    run: TuiSessionRun,
    *,
    source_count: int,
) -> dict[str, object]:
    return {
        "operation": "ask",
        "retrieval_mode": run.retrieval_mode,
        "retrieval_method": run.retrieval_method,
        "limit": run.limit,
        "max_context_chars": run.max_context_chars,
        "show_prompt": run.show_prompt,
        "trace_id": run.trace_id,
        "generation_status": run.generation_status,
        "generation_provider": run.generation_provider,
        "error": run.error,
        "source_count": source_count,
    }


def assistant_message_from_ask_result(result: AskPageState) -> TranscriptMessage:
    if result.error:
        sources = transcript_sources_from_search_results(result.sources)
        metadata = {
            "operation": "ask",
            "retrieval_mode": result.retrieval_mode,
            "retrieval_method": retrieval_method_from_sources(
                sources,
                result.retrieval_mode,
            ),
            "limit": result.limit,
            "max_context_chars": result.max_context_chars,
            "show_prompt": result.show_prompt,
            "generation_status": "failed",
            "generation_provider": result.generation_provider,
            "error": result.error,
            "source_count": len(sources),
            "trace_id": result.trace_id,
        }
        return TranscriptMessage(
            role="assistant",
            text=result.error,
            metadata=metadata,
            sources=sources,
        )

    messages = messages_from_ask_state(result)
    for message in messages:
        if message.role != "assistant":
            continue
        metadata = dict(message.metadata)
        metadata.setdefault(
            "retrieval_method",
            retrieval_method_from_sources(message.sources, result.retrieval_mode),
        )
        metadata.setdefault("limit", result.limit)
        metadata.setdefault("max_context_chars", result.max_context_chars)
        metadata.setdefault("show_prompt", result.show_prompt)
        metadata.setdefault("trace_id", result.trace_id)
        return replace(message, metadata=metadata)

    sources = transcript_sources_from_search_results(result.sources)
    metadata = {
        "operation": "ask",
        "retrieval_mode": result.retrieval_mode,
        "retrieval_method": retrieval_method_from_sources(
            sources,
            result.retrieval_mode,
        ),
        "limit": result.limit,
        "max_context_chars": result.max_context_chars,
        "show_prompt": result.show_prompt,
        "generation_status": result.generation_status,
        "generation_provider": result.generation_provider,
        "source_count": len(sources),
        "trace_id": result.trace_id,
    }
    return TranscriptMessage(
        role="assistant",
        text=result.status or "Ask completed.",
        metadata=metadata,
        sources=sources,
    )


def session_run_from_ask_result(
    result: AskPageState,
    message: TranscriptMessage,
) -> TuiSessionRun:
    metadata = message.metadata
    return TuiSessionRun(
        retrieval_mode=result.retrieval_mode,
        retrieval_method=str(
            metadata.get(
                "retrieval_method",
                retrieval_method_from_sources(
                    message.sources,
                    result.retrieval_mode,
                ),
            )
        ),
        limit=result.limit,
        max_context_chars=result.max_context_chars,
        show_prompt=result.show_prompt,
        trace_id=result.trace_id,
        generation_status=str(metadata.get("generation_status") or ""),
        generation_provider=(
            str(metadata["generation_provider"])
            if metadata.get("generation_provider") is not None
            else None
        ),
        error=(
            str(metadata["error"])
            if metadata.get("error") is not None
            else result.error
        ),
        prompt_preview=result.prompt_preview,
    )


def fallback_title_from_question(question: str) -> str:
    title = " ".join(question.split())
    if len(title) <= 80:
        return title
    return f"{title[:77].rstrip()}..."
