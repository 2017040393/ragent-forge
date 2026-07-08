from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, cast

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
_LATEX_INLINE_MATH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:\\\\|\\)\((.*?)(?:\\\\|\\)\)"
)
_LATEX_DISPLAY_MATH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:\\\\|\\)\[(.*?)(?:\\\\|\\)\]"
)
_DOLLAR_INLINE_MATH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\\)\$([^$\n]*?[\\^_=][^$\n]*?)(?<!\\)\$"
)
_LATEX_SCRIPT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<operator>[\^_])"
    r"(?:\{(?P<braced>[A-Za-z0-9+\-=()]{1,8})\}|"
    r"(?P<plain>[A-Za-z0-9+\-=()]))"
)
_STANDALONE_SUPERSCRIPT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<base>[A-Za-z0-9\u2102\u2115\u211a\u211d\u2124)\]])"
    r"\^(?:\{(?P<braced>[A-Za-z0-9+\-=()]{1,8})\}|"
    r"(?P<plain>[A-Za-z0-9+\-=()]))"
)
_LATEX_MATHBB_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:\\\\|\\)mathbb\{([A-Za-z])\}"
)
_LATEX_MATHBB_SYMBOLS: Final[dict[str, str]] = {
    "C": "\u2102",
    "N": "\u2115",
    "Q": "\u211a",
    "R": "\u211d",
    "Z": "\u2124",
}
_LATEX_SYMBOLS: Final[dict[str, str]] = {
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "delta": "\u03b4",
    "epsilon": "\u03b5",
    "varepsilon": "\u03b5",
    "theta": "\u03b8",
    "lambda": "\u03bb",
    "mu": "\u03bc",
    "pi": "\u03c0",
    "sigma": "\u03c3",
    "phi": "\u03c6",
    "varphi": "\u03c6",
    "omega": "\u03c9",
    "Gamma": "\u0393",
    "Delta": "\u0394",
    "Theta": "\u0398",
    "Lambda": "\u039b",
    "Pi": "\u03a0",
    "Sigma": "\u03a3",
    "Phi": "\u03a6",
    "Omega": "\u03a9",
    "le": "\u2264",
    "leq": "\u2264",
    "ge": "\u2265",
    "geq": "\u2265",
    "neq": "\u2260",
    "approx": "\u2248",
    "propto": "\u221d",
    "in": "\u2208",
    "notin": "\u2209",
    "subset": "\u2282",
    "subseteq": "\u2286",
    "supset": "\u2283",
    "supseteq": "\u2287",
    "cup": "\u222a",
    "cap": "\u2229",
    "times": "\u00d7",
    "cdot": "\u00b7",
    "pm": "\u00b1",
    "to": "\u2192",
    "rightarrow": "\u2192",
    "leftarrow": "\u2190",
    "Rightarrow": "\u21d2",
    "infty": "\u221e",
    "sum": "\u2211",
    "prod": "\u220f",
    "int": "\u222b",
    "forall": "\u2200",
    "exists": "\u2203",
    "nabla": "\u2207",
    "partial": "\u2202",
    "sqrt": "\u221a",
    "ldots": "...",
    "dots": "...",
}
_SUPERSCRIPT_CHARS: Final[dict[str, str]] = {
    "0": "\u2070",
    "1": "\u00b9",
    "2": "\u00b2",
    "3": "\u00b3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
    "+": "\u207a",
    "-": "\u207b",
    "=": "\u207c",
    "(": "\u207d",
    ")": "\u207e",
    "a": "\u1d43",
    "b": "\u1d47",
    "c": "\u1d9c",
    "d": "\u1d48",
    "e": "\u1d49",
    "f": "\u1da0",
    "g": "\u1d4d",
    "h": "\u02b0",
    "i": "\u2071",
    "j": "\u02b2",
    "k": "\u1d4f",
    "l": "\u02e1",
    "m": "\u1d50",
    "n": "\u207f",
    "o": "\u1d52",
    "p": "\u1d56",
    "r": "\u02b3",
    "s": "\u02e2",
    "t": "\u1d57",
    "u": "\u1d58",
    "v": "\u1d5b",
    "w": "\u02b7",
    "x": "\u02e3",
    "y": "\u02b8",
    "z": "\u1dbb",
}
_SUBSCRIPT_CHARS: Final[dict[str, str]] = {
    "0": "\u2080",
    "1": "\u2081",
    "2": "\u2082",
    "3": "\u2083",
    "4": "\u2084",
    "5": "\u2085",
    "6": "\u2086",
    "7": "\u2087",
    "8": "\u2088",
    "9": "\u2089",
    "+": "\u208a",
    "-": "\u208b",
    "=": "\u208c",
    "(": "\u208d",
    ")": "\u208e",
    "a": "\u2090",
    "e": "\u2091",
    "h": "\u2095",
    "i": "\u1d62",
    "j": "\u2c7c",
    "k": "\u2096",
    "l": "\u2097",
    "m": "\u2098",
    "n": "\u2099",
    "o": "\u2092",
    "p": "\u209a",
    "r": "\u1d63",
    "s": "\u209b",
    "t": "\u209c",
    "u": "\u1d64",
    "v": "\u1d65",
    "x": "\u2093",
}

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
_SOURCE_NAVIGATION_HINT = (
    "Use /source <rank>, /source next, or /source prev to inspect evidence."
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
    retrieval_mode: RetrievalMode = "hybrid"
    limit: int = 5
    max_context_chars: int = 4000
    show_prompt: bool = False
    running: bool = False
    notice: str | None = None
    messages: tuple[TranscriptMessage, ...] = ()
    selected_source: TranscriptSource | None = None
    available_sources: tuple[TranscriptSource, ...] = ()
    inspector_text: str | None = None


def create_initial_shell_state() -> ShellState:
    return ShellState()


def append_message(state: ShellState, message: TranscriptMessage) -> ShellState:
    notice = state.notice
    if message.role in {"tool", "error"}:
        notice = message.text
    elif message.role == "assistant":
        notice = None
    updated = replace(
        state,
        messages=(*state.messages, message),
        notice=notice,
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
        notice=None,
        messages=(),
        selected_source=None,
        available_sources=(),
        inspector_text=None,
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


def set_notice(state: ShellState, text: str | None) -> ShellState:
    return replace(state, notice=text)


def set_inspector_text(state: ShellState, text: str | None) -> ShellState:
    return replace(state, inspector_text=text)


def select_source(
    state: ShellState,
    source: TranscriptSource | None,
) -> ShellState:
    return replace(state, selected_source=source, inspector_text=None)


def set_available_sources(
    state: ShellState,
    sources: list[TranscriptSource] | tuple[TranscriptSource, ...],
) -> ShellState:
    available_sources = tuple(sources)
    return replace(
        state,
        available_sources=available_sources,
        selected_source=available_sources[0] if available_sources else None,
        inspector_text=None,
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
    lines = _display_text_for_message(message).splitlines() or [""]
    indented = "\n".join(f"  {line}" for line in lines)
    rendered = f"{heading}:\n{indented}"
    if message.sources:
        rendered = "\n\n".join([rendered, format_transcript_sources(message.sources)])
    return rendered


def format_transcript(
    messages: list[TranscriptMessage] | tuple[TranscriptMessage, ...],
) -> str:
    return "\n\n".join(format_transcript_message(message) for message in messages)


def format_conversation_transcript(
    messages: list[TranscriptMessage] | tuple[TranscriptMessage, ...],
) -> str:
    visible_messages = [
        replace(message, sources=())
        for message in messages
        if message.role in {"user", "assistant"}
    ]
    return "\n\n".join(
        format_transcript_message(message) for message in visible_messages
    )


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
    lines.extend(["", _SOURCE_NAVIGATION_HINT])
    return "\n".join(lines)


def format_shell_status(state: ShellState) -> str:
    prompt = "on" if state.show_prompt else "off"
    status = "running" if state.running else "idle"
    summary = (
        f"mode: {state.retrieval_mode} | "
        f"limit: {state.limit} | "
        f"context: {state.max_context_chars} | "
        f"prompt: {prompt} | "
        f"status: {status}"
    )
    if state.notice:
        return "\n".join([summary, state.notice.splitlines()[0]])
    return summary


def format_shell_inspector(state: ShellState) -> str:
    if state.inspector_text is not None:
        return _safe_display_text(state.inspector_text)

    if state.selected_source is not None:
        return format_shell_source_details(state.selected_source)

    if state.notice:
        return "\n".join(["Status", "", _safe_display_text(state.notice)])

    return "Inspector\n\nNo source selected."


def format_shell_source_details(source: TranscriptSource) -> str:
    preview = _truncate_tail(
        _safe_display_text(source.preview),
        _MAX_SOURCE_PREVIEW_LENGTH,
    )
    lines = [
        "Selected source",
        "",
        "Evidence",
        f"rank: {source.rank}",
        f"score: {source.score:.4g}",
        "",
        "Location",
        f"source: {compact_source_label(source.source_path, source.metadata)}",
        f"chunk: {compact_chunk_label(source.chunk_id)}",
    ]
    source_metadata_lines = _format_structured_source_metadata(source.metadata)
    if source_metadata_lines:
        lines.extend(source_metadata_lines)
    lines.extend(["", "Preview", "preview:"])
    lines.extend(f"  {line}" for line in preview.splitlines() or [""])

    metadata_lines = _format_retrieval_source_metadata(source.metadata)
    if metadata_lines:
        lines.extend(["", "Retrieval metadata", "", *metadata_lines])
    return "\n".join(lines)


def format_prompt_preview_inspector(prompt_preview: str) -> str:
    lines = ["Prompt preview", ""]
    lines.extend(prompt_preview.splitlines() or [""])
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


def _display_text_for_message(message: TranscriptMessage) -> str:
    text = _safe_display_text(message.text)
    if message.role == "assistant":
        return _normalize_latex_math_for_display(text)
    return text


def _normalize_latex_math_for_display(text: str) -> str:
    normalized_lines: list[str] = []
    in_code_block = False
    for line in text.splitlines(keepends=True):
        if _is_markdown_code_fence(line):
            normalized_lines.append(line)
            in_code_block = not in_code_block
            continue
        if in_code_block:
            normalized_lines.append(line)
        else:
            normalized_lines.append(_normalize_latex_math_segment(line))
    return "".join(normalized_lines)


def _is_markdown_code_fence(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _normalize_latex_math_segment(text: str) -> str:
    text = _LATEX_DISPLAY_MATH_PATTERN.sub(_replace_latex_math_match, text)
    text = _LATEX_INLINE_MATH_PATTERN.sub(_replace_latex_math_match, text)
    text = _DOLLAR_INLINE_MATH_PATTERN.sub(_replace_latex_math_match, text)
    text = _normalize_latex_symbols(text)
    return _normalize_standalone_superscripts(text)


def _replace_latex_math_match(match: re.Match[str]) -> str:
    return _normalize_latex_math_content(match.group(1).strip())


def _normalize_latex_math_content(text: str) -> str:
    return _normalize_latex_scripts(_normalize_latex_symbols(text))


def _normalize_latex_scripts(text: str) -> str:
    return _LATEX_SCRIPT_PATTERN.sub(_replace_latex_script_match, text)


def _replace_latex_script_match(match: re.Match[str]) -> str:
    operator = match.group("operator")
    raw_value = match.group("braced") or match.group("plain") or ""
    table = _SUPERSCRIPT_CHARS if operator == "^" else _SUBSCRIPT_CHARS
    converted = _translate_script_text(raw_value, table)
    if converted is None:
        return match.group(0)
    return converted


def _normalize_standalone_superscripts(text: str) -> str:
    return _STANDALONE_SUPERSCRIPT_PATTERN.sub(
        _replace_standalone_superscript_match,
        text,
    )


def _replace_standalone_superscript_match(match: re.Match[str]) -> str:
    raw_value = match.group("braced") or match.group("plain") or ""
    converted = _translate_script_text(raw_value, _SUPERSCRIPT_CHARS)
    if converted is None:
        return match.group(0)
    return f"{match.group('base')}{converted}"


def _translate_script_text(text: str, table: dict[str, str]) -> str | None:
    chars: list[str] = []
    for char in text:
        mapped = table.get(char)
        if mapped is None:
            return None
        chars.append(mapped)
    return "".join(chars)


def _normalize_latex_symbols(text: str) -> str:
    text = _LATEX_MATHBB_PATTERN.sub(_replace_latex_mathbb_match, text)
    for command, symbol in _LATEX_SYMBOLS.items():
        text = re.sub(rf"(?:\\\\|\\){command}(?![A-Za-z])", symbol, text)
    return text


def _replace_latex_mathbb_match(match: re.Match[str]) -> str:
    value = match.group(1)
    return _LATEX_MATHBB_SYMBOLS.get(value, value)


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
