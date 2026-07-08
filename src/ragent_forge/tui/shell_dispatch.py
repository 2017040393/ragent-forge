from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, cast

from ragent_forge.tui.commands import parse_tui_input
from ragent_forge.tui.shell_models import (
    ShellState,
    clear_transcript,
    format_selected_source_ack,
    select_next_source,
    select_previous_source,
    select_source_by_rank,
    set_limit,
    set_max_context_chars,
    set_notice,
    set_retrieval_mode,
    set_show_prompt,
)

ShellAction = Literal["none", "quit", "search", "ask", "sources", "help"]
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

_READ_ONLY_SUCCESS_MESSAGES = {
    "docs": "Document summary loaded in Inspector.",
    "trace": "Trace summary loaded in Inspector.",
    "settings": "Settings summary loaded in Inspector.",
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
        return ShellDispatchResult(set_notice(state, parsed.error))

    if parsed.name == "ask":
        return ShellDispatchResult(state, action="ask", ask_question=parsed.args)
    if parsed.name == "help":
        return ShellDispatchResult(state, action="help")
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
        return _apply_sources_command(state)
    if parsed.name == "source":
        return ShellDispatchResult(_apply_source_command(state, parsed.args))
    if parsed.name in {"docs", "trace", "settings"}:
        return ShellDispatchResult(
            _apply_read_only_command(
                state,
                cast(Literal["docs", "trace", "settings"], parsed.name),
                handlers,
            )
        )
    if parsed.name in _PLANNED_NOT_WIRED_MESSAGES:
        return ShellDispatchResult(
            set_notice(
                state,
                _PLANNED_NOT_WIRED_MESSAGES[parsed.name],
            )
        )
    if parsed.name == "exit":
        return ShellDispatchResult(state, action="quit")

    return ShellDispatchResult(set_notice(state, "Unknown command."))


def _apply_mode_command(state: ShellState, mode: str) -> ShellState:
    try:
        updated = set_retrieval_mode(state, mode)
    except ValueError as exc:
        return set_notice(state, str(exc))
    return set_notice(
        updated,
        f"retrieval mode set to {updated.retrieval_mode}",
    )


def _apply_limit_command(state: ShellState, value: str) -> ShellState:
    try:
        limit = int(value)
    except ValueError:
        return set_notice(state, f"Invalid limit: {value}")

    try:
        updated = set_limit(state, limit)
    except ValueError as exc:
        return set_notice(state, str(exc))
    return set_notice(updated, f"limit set to {updated.limit}")


def _apply_context_command(state: ShellState, value: str) -> ShellState:
    try:
        max_context_chars = int(value)
    except ValueError:
        return set_notice(
            state,
            f"Invalid context value: {value}",
        )

    try:
        updated = set_max_context_chars(state, max_context_chars)
    except ValueError as exc:
        return set_notice(state, str(exc))
    return set_notice(
        updated,
        f"max context chars set to {updated.max_context_chars}",
    )


def _apply_prompt_command(state: ShellState, value: str) -> ShellState:
    normalized = value.lower()
    if normalized not in {"on", "off"}:
        return set_notice(state, "Usage: /prompt on|off")

    enabled = normalized == "on"
    updated = set_show_prompt(state, enabled)
    status = "enabled" if enabled else "disabled"
    return set_notice(updated, f"prompt preview {status}")


def _apply_sources_command(state: ShellState) -> ShellDispatchResult:
    if not state.available_sources:
        return ShellDispatchResult(set_notice(state, NO_SOURCES_MESSAGE))
    return ShellDispatchResult(state, action="sources")


def _apply_source_command(state: ShellState, value: str) -> ShellState:
    normalized = value.lower()
    if normalized == "next":
        return _select_source_with_ack(state, select_next_source)
    if normalized == "prev":
        return _select_source_with_ack(state, select_previous_source)

    try:
        rank = int(value)
    except ValueError:
        return set_notice(state, "Usage: /source <rank|next|prev>")

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
        return set_notice(state, message)

    if updated.selected_source is None:
        return set_notice(updated, NO_SOURCES_MESSAGE)
    return set_notice(
        updated,
        format_selected_source_ack(updated.selected_source),
    )


def _apply_read_only_command(
    state: ShellState,
    command: Literal["docs", "trace", "settings"],
    handlers: ShellReadOnlyHandlers | None,
) -> ShellState:
    handler = _read_only_handler(command, handlers)
    if handler is None:
        return set_notice(
            state,
            _PLANNED_NOT_WIRED_MESSAGES[command],
        )

    try:
        text = handler()
    except Exception:
        return set_notice(
            state,
            _READ_ONLY_ERROR_MESSAGES[command],
        )

    updated = replace(state, inspector_text=text)
    return set_notice(updated, _READ_ONLY_SUCCESS_MESSAGES[command])


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
