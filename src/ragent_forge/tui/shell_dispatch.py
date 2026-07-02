from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ragent_forge.tui.commands import format_tui_command_help, parse_tui_input
from ragent_forge.tui.shell_models import (
    ShellState,
    TranscriptMessage,
    TranscriptRole,
    append_message,
    append_messages,
    clear_transcript,
    set_limit,
    set_max_context_chars,
    set_retrieval_mode,
    set_show_prompt,
)

ShellAction = Literal["none", "quit"]

ASK_NOT_WIRED_MESSAGE = (
    "Ask execution from Shell is not wired yet. Use the Ask page for now."
)

_PLANNED_NOT_WIRED_MESSAGES = {
    "search": "/search dispatch is not wired yet. Use the Search page for now.",
    "docs": "/docs dispatch is not wired yet. Use the Documents page for now.",
    "trace": "/trace dispatch is not wired yet. Use the Trace page for now.",
    "settings": (
        "/settings dispatch is not wired yet. Use the Settings page for now."
    ),
}


@dataclass(frozen=True)
class ShellDispatchResult:
    state: ShellState
    action: ShellAction = "none"


def apply_shell_input(state: ShellState, text: str) -> ShellDispatchResult:
    parsed = parse_tui_input(text)
    if parsed.error:
        return ShellDispatchResult(_append_shell_message(state, "error", parsed.error))

    if parsed.name == "ask":
        return ShellDispatchResult(_append_ask_placeholder(state, parsed.args))
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


def _append_ask_placeholder(state: ShellState, question: str) -> ShellState:
    return append_messages(
        state,
        (
            TranscriptMessage(role="user", text=question),
            TranscriptMessage(role="tool", text=ASK_NOT_WIRED_MESSAGE),
        ),
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


def _append_shell_message(
    state: ShellState,
    role: TranscriptRole,
    text: str,
) -> ShellState:
    return append_message(state, TranscriptMessage(role=role, text=text))
