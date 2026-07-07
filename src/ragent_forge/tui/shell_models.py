from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from ragent_forge.app.services.chunk_service import make_preview
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.source_labels import (
    format_source_metadata as _format_structured_source_metadata,
)
from ragent_forge.tui.view_models import (
    AskPageState,
    SearchPageState,
    compact_chunk_label,
    compact_source_label,
)

TranscriptRole = Literal["system", "user", "assistant", "tool", "error"]
RetrievalMode = Literal["lexical", "bm25", "semantic", "hybrid"]

WELCOME_MESSAGE = (
    "RAGentForge command shell.\n"
    "Type a question to ask your local knowledge base, or type /help for commands."
)

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"\bapi_key\s*[:=]", re.IGNORECASE),
    re.compile(r"\bauthorization\s*:", re.IGNORECASE),
    re.compile(r"^\s*bearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bsecret\s*[:=]", re.IGNORECASE),
    re.compile(r"\btoken\s*[:=]", re.IGNORECASE),
)

_SOURCE_METADATA_LABELS = {
    "retrieval_method": "method",
    "fusion_method": "fusion",
    "matched_modes": "matched",
    "sparse_method": "sparse_method",
    "dense_method": "dense_method",
    "sparse_rank": "sparse_rank",
    "dense_rank": "dense_rank",
    "sparse_score": "sparse_score",
    "dense_score": "dense_score",
    "hybrid_score": "hybrid_score",
    "sparse_weight": "sparse_weight",
    "dense_weight": "dense_weight",
}

_MAX_SOURCE_LABEL_WIDTH = 40
_MAX_SOURCE_PREVIEW_LENGTH = 240


@dataclass(frozen=True)
class TranscriptSource:
    rank: int
    chunk_id: str
    source_path: str
    score: float
    preview: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptMessage:
    role: TranscriptRole
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sources: tuple[TranscriptSource, ...] = ()


@dataclass(frozen=True)
class ShellState:
    retrieval_mode: RetrievalMode = "lexical"
    limit: int = 5
    max_context_chars: int = 4000
    show_prompt: bool = False
    running: bool = False
    messages: tuple[TranscriptMessage, ...] = ()
    selected_source: TranscriptSource | None = None
    available_sources: tuple[TranscriptSource, ...] = ()


def create_initial_shell_state() -> ShellState:
    return ShellState(messages=(_welcome_message(),))


def append_message(state: ShellState, message: TranscriptMessage) -> ShellState:
    updated = replace(
        state,
        messages=(*state.messages, message),
    )
    if message.sources:
        return set_available_sources(updated, message.sources)
    return updated


def append_messages(
    state: ShellState,
    messages: list[TranscriptMessage] | tuple[TranscriptMessage, ...],
) -> ShellState:
    updated = state
    for message in messages:
        updated = append_message(updated, message)
    return updated


def clear_transcript(state: ShellState) -> ShellState:
    return replace(
        state,
        running=False,
        messages=(_welcome_message(),),
        selected_source=None,
        available_sources=(),
    )


def set_retrieval_mode(state: ShellState, mode: str) -> ShellState:
    if mode not in {"lexical", "bm25", "semantic", "hybrid"}:
        raise ValueError(f"Invalid retrieval mode: {mode}")
    return replace(state, retrieval_mode=cast(RetrievalMode, mode))


def set_limit(state: ShellState, limit: int) -> ShellState:
    if limit <= 0:
        raise ValueError("Limit must be positive.")
    return replace(state, limit=limit)


def set_max_context_chars(
    state: ShellState,
    max_context_chars: int,
) -> ShellState:
    if max_context_chars <= 0:
        raise ValueError("Max context chars must be positive.")
    return replace(state, max_context_chars=max_context_chars)


def set_show_prompt(state: ShellState, show_prompt: bool) -> ShellState:
    return replace(state, show_prompt=show_prompt)


def set_running(state: ShellState, running: bool) -> ShellState:
    return replace(state, running=running)


def select_source(
    state: ShellState,
    source: TranscriptSource | None,
) -> ShellState:
    return replace(state, selected_source=source)


def set_available_sources(
    state: ShellState,
    sources: list[TranscriptSource] | tuple[TranscriptSource, ...],
) -> ShellState:
    available_sources = tuple(sources)
    return replace(
        state,
        available_sources=available_sources,
        selected_source=available_sources[0] if available_sources else None,
    )


def select_source_by_rank(state: ShellState, rank: int) -> ShellState:
    if not state.available_sources:
        raise ValueError(
            "No sources available. Run /search <query> or ask a question first."
        )
    if rank <= 0:
        raise ValueError("Source rank must be a positive integer.")
    if rank > len(state.available_sources):
        raise ValueError(
            "Source rank out of range. "
            f"Available sources: 1-{len(state.available_sources)}."
        )
    return select_source(state, state.available_sources[rank - 1])


def select_next_source(state: ShellState) -> ShellState:
    if not state.available_sources:
        raise ValueError(
            "No sources available. Run /search <query> or ask a question first."
        )
    selected_index = _selected_source_index(state)
    next_index = (
        0
        if selected_index is None
        else (selected_index + 1) % len(state.available_sources)
    )
    return select_source(state, state.available_sources[next_index])


def select_previous_source(state: ShellState) -> ShellState:
    if not state.available_sources:
        raise ValueError(
            "No sources available. Run /search <query> or ask a question first."
        )
    selected_index = _selected_source_index(state)
    previous_index = (
        0
        if selected_index is None
        else (selected_index - 1) % len(state.available_sources)
    )
    return select_source(state, state.available_sources[previous_index])


def format_selected_source_ack(source: TranscriptSource) -> str:
    return (
        f"selected source {source.rank}: "
        f"{compact_source_label(source.source_path, source.metadata)}"
    )


def _selected_source_index(state: ShellState) -> int | None:
    if state.selected_source is None:
        return None
    for index, source in enumerate(state.available_sources):
        if source == state.selected_source:
            return index
    return None


def transcript_sources_from_search_results(
    results: list[SearchResult] | tuple[SearchResult, ...],
) -> tuple[TranscriptSource, ...]:
    return tuple(
        TranscriptSource(
            rank=index,
            chunk_id=result.chunk_id,
            source_path=result.source_path,
            score=result.score,
            preview=make_preview(result.text, max_length=120),
            metadata=_safe_metadata(result.metadata),
        )
        for index, result in enumerate(results, start=1)
    )


def messages_from_ask_state(state: AskPageState) -> tuple[TranscriptMessage, ...]:
    if state.error:
        return (
            TranscriptMessage(
                role="error",
                text=state.error,
                metadata={
                    "operation": "ask",
                    "retrieval_mode": state.retrieval_mode,
                },
            ),
        )

    sources = transcript_sources_from_search_results(state.sources)
    if state.answer:
        return (
            TranscriptMessage(
                role="assistant",
                text=state.answer,
                metadata=_ask_metadata(state),
                sources=sources,
            ),
        )

    if state.status:
        return (
            TranscriptMessage(
                role="tool",
                text=state.status,
                metadata=_ask_metadata(state),
                sources=sources,
            ),
        )

    return (
        TranscriptMessage(
            role="tool",
            text="Ask completed.",
            metadata={
                "operation": "ask",
                "retrieval_mode": state.retrieval_mode,
            },
            sources=sources,
        ),
    )


def message_from_search_results(
    query: str,
    retrieval_mode: str,
    results: list[SearchResult] | tuple[SearchResult, ...],
) -> TranscriptMessage:
    sources = transcript_sources_from_search_results(results)
    result_count = len(results)
    text = (
        f"Search results for: {query}\n"
        f"Results: {result_count} | mode: {retrieval_mode}"
        if result_count
        else "No matches found. Try another query or retrieval mode."
    )
    return TranscriptMessage(
        role="tool",
        text=text,
        metadata={
            "operation": "search",
            "query": query,
            "retrieval_mode": retrieval_mode,
            "result_count": result_count,
        },
        sources=sources,
    )


def message_from_search_state(state: SearchPageState) -> TranscriptMessage:
    if state.error:
        return TranscriptMessage(
            role="error",
            text=state.error,
            metadata={
                "operation": "search",
                "query": state.query,
                "retrieval_mode": state.retrieval_mode,
                "limit": state.limit,
            },
        )

    sources = transcript_sources_from_search_results(state.results)
    result_count = len(state.results)
    text = (
        f"Search results for: {state.query}\n"
        f"Results: {result_count} | mode: {state.retrieval_mode}"
        if result_count
        else "No matches found. Try another query or retrieval mode."
    )
    return TranscriptMessage(
        role="tool",
        text=text,
        metadata={
            "operation": "search",
            "query": state.query,
            "retrieval_mode": state.retrieval_mode,
            "limit": state.limit,
            "result_count": result_count,
        },
        sources=sources,
    )


def format_transcript_message(message: TranscriptMessage) -> str:
    heading = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
        "error": "Error",
    }[message.role]
    lines = _safe_display_text(message.text).splitlines() or [""]
    indented = "\n".join(f"  {line}" for line in lines)
    rendered = f"{heading}:\n{indented}"
    if message.sources:
        rendered = "\n\n".join([rendered, format_transcript_sources(message.sources)])
    return rendered


def format_transcript(
    messages: list[TranscriptMessage] | tuple[TranscriptMessage, ...],
) -> str:
    return "\n\n".join(format_transcript_message(message) for message in messages)


def format_transcript_sources(
    sources: list[TranscriptSource] | tuple[TranscriptSource, ...],
) -> str:
    if not sources:
        return "Sources:\nNo sources."

    labels = [
        _truncate_tail(
            compact_source_label(source.source_path, source.metadata),
            _MAX_SOURCE_LABEL_WIDTH,
        )
        for source in sources
    ]
    label_width = max(len(label) for label in labels)
    lines = ["Sources:"]
    lines.extend(
        (
            f"{source.rank}. {label:<{label_width}}  "
            f"score={source.score:.4g}  "
            f"chunk={compact_chunk_label(source.chunk_id)}"
        )
        for source, label in zip(sources, labels, strict=True)
    )
    return "\n".join(lines)


def format_shell_status(state: ShellState) -> str:
    prompt = "on" if state.show_prompt else "off"
    status = "running" if state.running else "idle"
    return (
        f"mode: {state.retrieval_mode} | "
        f"limit: {state.limit} | "
        f"context: {state.max_context_chars} | "
        f"prompt: {prompt} | "
        f"status: {status}"
    )


def format_shell_inspector(state: ShellState) -> str:
    prompt = "on" if state.show_prompt else "off"
    selected_source = (
        compact_source_label(
            state.selected_source.source_path,
            state.selected_source.metadata,
        )
        if state.selected_source is not None
        else "none"
    )
    lines = [
        "Shell details",
        "",
        f"mode: {state.retrieval_mode}",
        f"limit: {state.limit}",
        f"context: {state.max_context_chars}",
        f"prompt: {prompt}",
        f"messages: {len(state.messages)}",
        f"selected source: {selected_source}",
    ]
    if state.selected_source is not None:
        lines.extend(["", format_shell_source_details(state.selected_source)])
    return "\n".join(lines)


def format_shell_source_details(source: TranscriptSource) -> str:
    preview = _truncate_tail(
        _safe_display_text(source.preview),
        _MAX_SOURCE_PREVIEW_LENGTH,
    )
    lines = [
        "Selected source",
        "",
        f"rank: {source.rank}",
        f"source: {compact_source_label(source.source_path, source.metadata)}",
        f"chunk: {compact_chunk_label(source.chunk_id)}",
        f"score: {source.score:.4g}",
    ]
    source_metadata_lines = _format_structured_source_metadata(source.metadata)
    if source_metadata_lines:
        lines.extend(source_metadata_lines)
    lines.extend(["", "preview:"])
    lines.extend(f"  {line}" for line in preview.splitlines() or [""])

    metadata_lines = _format_retrieval_source_metadata(source.metadata)
    if metadata_lines:
        lines.extend(["", "Retrieval metadata", "", *metadata_lines])
    return "\n".join(lines)


def _format_retrieval_source_metadata(metadata: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, label in _SOURCE_METADATA_LABELS.items():
        if key in metadata:
            lines.append(f"{label}: {_format_source_metadata_value(metadata[key])}")
    return lines


def _format_source_metadata_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def _truncate_tail(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return "." * max_length
    return f"{text[: max_length - 3]}..."


def _welcome_message() -> TranscriptMessage:
    return TranscriptMessage(role="system", text=WELCOME_MESSAGE)


def _ask_metadata(state: AskPageState) -> dict[str, Any]:
    return {
        "operation": "ask",
        "retrieval_mode": state.retrieval_mode,
        "generation_status": state.generation_status,
        "generation_provider": state.generation_provider,
        "source_count": len(state.sources),
    }


def _safe_display_text(text: str) -> str:
    sanitized_lines: list[str] = []
    for line in text.splitlines():
        if _looks_like_sensitive_text(line):
            sanitized_lines.append("<hidden>")
        else:
            sanitized_lines.append(line)
    return "\n".join(sanitized_lines)


def _looks_like_sensitive_text(line: str) -> bool:
    return any(pattern.search(line) for pattern in _SENSITIVE_TEXT_PATTERNS)


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sensitive_fragments = (
        "api_key",
        "secret",
        "token",
        "authorization",
        "embedding",
        "embeddings",
        "vector",
    )
    return {
        key: value
        for key, value in metadata.items()
        if not any(fragment in key.lower() for fragment in sensitive_fragments)
    }
