from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from ragent_forge.app.services.chunk_service import make_preview
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.tui.view_models import (
    AskPageState,
    compact_chunk_label,
    compact_source_label,
)

TranscriptRole = Literal["system", "user", "assistant", "tool", "error"]
RetrievalMode = Literal["lexical", "semantic", "hybrid"]

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


def create_initial_shell_state() -> ShellState:
    return ShellState(messages=(_welcome_message(),))


def append_message(state: ShellState, message: TranscriptMessage) -> ShellState:
    selected_source = state.selected_source
    if selected_source is None and message.sources:
        selected_source = message.sources[0]
    return replace(
        state,
        messages=(*state.messages, message),
        selected_source=selected_source,
    )


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
    )


def set_retrieval_mode(state: ShellState, mode: str) -> ShellState:
    if mode not in {"lexical", "semantic", "hybrid"}:
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
    return f"{heading}:\n{indented}"


def format_transcript(
    messages: list[TranscriptMessage] | tuple[TranscriptMessage, ...],
) -> str:
    return "\n\n".join(format_transcript_message(message) for message in messages)


def format_transcript_sources(
    sources: list[TranscriptSource] | tuple[TranscriptSource, ...],
) -> str:
    if not sources:
        return "Sources:\nNo sources."

    lines = ["Sources:"]
    lines.extend(
        (
            f"{source.rank}. {compact_source_label(source.source_path)}      "
            f"score={source.score:.4g}  "
            f"chunk={compact_chunk_label(source.chunk_id)}"
        )
        for source in sources
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
