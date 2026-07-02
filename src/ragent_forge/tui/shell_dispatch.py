from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from ragent_forge.tui.commands import format_tui_command_help, parse_tui_input
from ragent_forge.tui.shell_models import (
    ShellState,
    TranscriptMessage,
    TranscriptRole,
    append_message,
    clear_transcript,
    format_selected_source_ack,
    select_next_source,
    select_previous_source,
    select_source_by_rank,
    set_limit,
    set_max_context_chars,
    set_retrieval_mode,
    set_show_prompt,
)

ShellAction = Literal["none", "quit", "search", "ask"]
NO_SOURCES_MESSAGE = (
    "No sources available. Run /search <query> or ask a question first."
)

_PLANNED_NOT_WIRED_MESSAGES = {
    "docs": "/docs is unavailable in this shell context.",
    "trace": "/trace is unavailable in this shell context.",
    "settings": "/settings is unavailable in this shell context.",
}

_READ_ONLY_ERROR_MESSAGES = {
    "docs": "Unable to load document summary.",
    "trace": "Unable to load trace summary.",
    "settings": "Unable to load settings summary.",
}


@dataclass(frozen=True)
class ShellReadOnlyHandlers:
    docs: Callable[[], str] | None = None
    trace: Callable[[], str] | None = None
    settings: Callable[[], str] | None = None


@dataclass(frozen=True)
class ShellDispatchResult:
    state: ShellState
    action: ShellAction = "none"
    search_query: str | None = None
    ask_question: str | None = None


def apply_shell_input(
    state: ShellState,
    text: str,
    *,
    handlers: ShellReadOnlyHandlers | None = None,
) -> ShellDispatchResult:
    parsed = parse_tui_input(text)
    if parsed.error:
        return ShellDispatchResult(_append_shell_message(state, "error", parsed.error))

    if parsed.name == "ask":
        return ShellDispatchResult(state, action="ask", ask_question=parsed.args)
    if parsed.name == "help":
        return ShellDispatchResult(
            _append_shell_message(state, "tool", format_tui_command_help())
        )
    if parsed.name == "clear":
        return ShellDispatchResult(clear_transcript(state))
    if parsed.name == "mode":
        return ShellDispatchResult(_apply_mode_command(state, parsed.args))
    if parsed.name == "limit":
        return ShellDispatchResult(_apply_limit_command(state, parsed.args))
    if parsed.name == "context":
        return ShellDispatchResult(_apply_context_command(state, parsed.args))
    if parsed.name == "prompt":
        return ShellDispatchResult(_apply_prompt_command(state, parsed.args))
    if parsed.name == "search":
        return ShellDispatchResult(state, action="search", search_query=parsed.args)
    if parsed.name == "sources":
        return ShellDispatchResult(_apply_sources_command(state))
    if parsed.name == "source":
        return ShellDispatchResult(_apply_source_command(state, parsed.args))
    if parsed.name in {"docs", "trace", "settings"}:
        return ShellDispatchResult(
            _apply_read_only_command(state, parsed.name, handlers)
        )
    if parsed.name in _PLANNED_NOT_WIRED_MESSAGES:
        return ShellDispatchResult(
            _append_shell_message(
                state,
                "tool",
                _PLANNED_NOT_WIRED_MESSAGES[parsed.name],
            )
        )
    if parsed.name == "exit":
        return ShellDispatchResult(state, action="quit")

    return ShellDispatchResult(
        _append_shell_message(state, "error", "Unknown command.")
    )


def _apply_mode_command(state: ShellState, mode: str) -> ShellState:
    try:
        updated = set_retrieval_mode(state, mode)
    except ValueError as exc:
        return _append_shell_message(state, "error", str(exc))
    return _append_shell_message(
        updated,
        "tool",
        f"retrieval mode set to {updated.retrieval_mode}",
    )


def _apply_limit_command(state: ShellState, value: str) -> ShellState:
    try:
        limit = int(value)
    except ValueError:
        return _append_shell_message(state, "error", f"Invalid limit: {value}")

    try:
        updated = set_limit(state, limit)
    except ValueError as exc:
        return _append_shell_message(state, "error", str(exc))
    return _append_shell_message(updated, "tool", f"limit set to {updated.limit}")


def _apply_context_command(state: ShellState, value: str) -> ShellState:
    try:
        max_context_chars = int(value)
    except ValueError:
        return _append_shell_message(
            state,
            "error",
            f"Invalid context value: {value}",
        )

    try:
        updated = set_max_context_chars(state, max_context_chars)
    except ValueError as exc:
        return _append_shell_message(state, "error", str(exc))
    return _append_shell_message(
        updated,
        "tool",
        f"max context chars set to {updated.max_context_chars}",
    )


def _apply_prompt_command(state: ShellState, value: str) -> ShellState:
    normalized = value.lower()
    if normalized not in {"on", "off"}:
        return _append_shell_message(state, "error", "Usage: /prompt on|off")

    enabled = normalized == "on"
    updated = set_show_prompt(state, enabled)
    status = "enabled" if enabled else "disabled"
    return _append_shell_message(updated, "tool", f"prompt preview {status}")


def _apply_sources_command(state: ShellState) -> ShellState:
    if not state.available_sources:
        return _append_shell_message(state, "tool", NO_SOURCES_MESSAGE)
    selected_source = state.selected_source
    available_sources = state.available_sources
    updated = append_message(
        state,
        TranscriptMessage(
            role="tool",
            text="Current sources",
            metadata={"operation": "source_list"},
            sources=available_sources,
        ),
    )
    return replace(
        updated,
        selected_source=selected_source,
        available_sources=available_sources,
    )


def _apply_source_command(state: ShellState, value: str) -> ShellState:
    normalized = value.lower()
    if normalized == "next":
        return _select_source_with_ack(state, select_next_source)
    if normalized == "prev":
        return _select_source_with_ack(state, select_previous_source)

    try:
        rank = int(value)
    except ValueError:
        return _append_shell_message(state, "error", "Usage: /source <rank|next|prev>")

    return _select_source_with_ack(
        state,
        lambda current: select_source_by_rank(current, rank),
    )


def _select_source_with_ack(
    state: ShellState,
    select_source_fn: Callable[[ShellState], ShellState],
) -> ShellState:
    try:
        updated = select_source_fn(state)
    except ValueError as exc:
        message = str(exc)
        role: TranscriptRole = "tool" if message == NO_SOURCES_MESSAGE else "error"
        return _append_shell_message(state, role, message)

    if updated.selected_source is None:
        return _append_shell_message(updated, "tool", NO_SOURCES_MESSAGE)
    return _append_shell_message(
        updated,
        "tool",
        format_selected_source_ack(updated.selected_source),
    )


def _apply_read_only_command(
    state: ShellState,
    command: Literal["docs", "trace", "settings"],
    handlers: ShellReadOnlyHandlers | None,
) -> ShellState:
    handler = _read_only_handler(command, handlers)
    if handler is None:
        return _append_shell_message(
            state,
            "tool",
            _PLANNED_NOT_WIRED_MESSAGES[command],
        )

    try:
        text = handler()
    except Exception:
        return _append_shell_message(
            state,
            "error",
            _READ_ONLY_ERROR_MESSAGES[command],
        )

    return append_message(
        state,
        TranscriptMessage(
            role="tool",
            text=text,
            metadata={"operation": "shell_command", "command": command},
        ),
    )


def _read_only_handler(
    command: Literal["docs", "trace", "settings"],
    handlers: ShellReadOnlyHandlers | None,
) -> Callable[[], str] | None:
    if handlers is None:
        return None
    if command == "docs":
        return handlers.docs
    if command == "trace":
        return handlers.trace
    return handlers.settings


def _append_shell_message(
    state: ShellState,
    role: TranscriptRole,
    text: str,
) -> ShellState:
    return append_message(state, TranscriptMessage(role=role, text=text))
