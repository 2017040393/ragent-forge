from __future__ import annotations

import re
from typing import Final, Literal

from rich.text import Text

TuiTextContext = Literal["status", "transcript", "inspector", "suggestions"]

ROLE_STYLES: Final[dict[str, str]] = {
    "System:": "bold bright_blue",
    "User:": "bold cyan",
    "Assistant:": "bold green",
    "Tool:": "bold yellow",
    "Error:": "bold red",
}
MODE_STYLES: Final[dict[str, str]] = {
    "lexical": "bright_blue",
    "bm25": "bright_magenta",
    "semantic": "cyan",
    "hybrid": "bold green",
}
METHOD_STYLES: Final[dict[str, str]] = {
    "lexical_token_overlap": "bright_blue",
    "bm25": "bright_magenta",
    "semantic_cosine_similarity": "cyan",
    "hybrid_rrf": "bold green",
}
POSITIVE_WORDS: Final[tuple[str, ...]] = ("idle", "ready", "success", "on")
WARNING_WORDS: Final[tuple[str, ...]] = ("running", "not configured")
NEGATIVE_WORDS: Final[tuple[str, ...]] = ("error", "failed", "missing")

_KEY_PATTERN = re.compile(r"(?m)^([A-Za-z][A-Za-z _-]*:)")
_SECTION_PATTERN = re.compile(
    r"(?m)^((?:Shell details|Selected source|Retrieval metadata|"
    r"Workspace|Ingest|Files|Semantic index|Recent chunks|"
    r"Search result|Ask source|Trace details|Latest trace|Steps|"
    r"Evidence|Location|Preview|Prompt preview|Metadata|Suggestions|Sources):?)$"
)
_SOURCE_RANK_PATTERN = re.compile(r"(?m)^(\d+\.)")
_SCORE_PATTERN = re.compile(r"\bscore=[^\s]+")
_NUMBER_PATTERN = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?(?![\w.-])")
_COMMAND_PATTERN = re.compile(r"/[A-Za-z][\w-]*")
_SELECTED_SUGGESTION_PATTERN = re.compile(r"(?m)^> [^\n]+")


def style_tui_text(text: str, context: TuiTextContext) -> Text:
    if context == "status":
        return style_shell_status(text)
    if context == "transcript":
        return style_transcript(text)
    if context == "inspector":
        return style_inspector(text)
    return style_command_suggestions(text)


def style_shell_status(text: str) -> Text:
    styled = Text(text)
    _style_keys(styled)
    _style_modes_and_methods(styled)
    _style_state_words(styled)
    _style_numbers(styled)
    return styled


def style_transcript(text: str) -> Text:
    styled = Text(text)
    _style_role_headings(styled)
    _style_sections(styled)
    _style_modes_and_methods(styled)
    _style_source_lines(styled)
    _style_commands(styled)
    return styled


def style_inspector(text: str) -> Text:
    styled = Text(text)
    _style_sections(styled)
    _style_keys(styled)
    _style_modes_and_methods(styled)
    _style_state_words(styled)
    _style_source_lines(styled)
    _style_numbers(styled)
    return styled


def style_command_suggestions(text: str) -> Text:
    styled = Text(text)
    _style_sections(styled)
    _style_commands(styled)
    _style_pattern(styled, _SELECTED_SUGGESTION_PATTERN, "bold reverse")
    return styled


def _style_role_headings(text: Text) -> None:
    for heading, style in ROLE_STYLES.items():
        _style_literal(text, heading, style)


def _style_sections(text: Text) -> None:
    _style_pattern(text, _SECTION_PATTERN, "bold cyan")


def _style_keys(text: Text) -> None:
    _style_pattern(text, _KEY_PATTERN, "dim")


def _style_modes_and_methods(text: Text) -> None:
    for method, style in METHOD_STYLES.items():
        _style_literal(text, method, style)
    for mode, style in MODE_STYLES.items():
        _style_pattern(text, re.compile(rf"\b{re.escape(mode)}\b"), style)


def _style_state_words(text: Text) -> None:
    for word in POSITIVE_WORDS:
        _style_pattern(text, re.compile(rf"\b{re.escape(word)}\b"), "green")
    for word in WARNING_WORDS:
        _style_pattern(text, re.compile(rf"\b{re.escape(word)}\b"), "bold yellow")
    for word in NEGATIVE_WORDS:
        _style_pattern(
            text,
            re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE),
            "bold red",
        )
    _style_pattern(text, re.compile(r"\boff\b"), "dim")


def _style_source_lines(text: Text) -> None:
    _style_pattern(text, _SOURCE_RANK_PATTERN, "bold cyan")
    _style_pattern(text, _SCORE_PATTERN, "yellow")


def _style_commands(text: Text) -> None:
    _style_pattern(text, _COMMAND_PATTERN, "cyan")


def _style_numbers(text: Text) -> None:
    _style_pattern(text, _NUMBER_PATTERN, "bright_cyan")


def _style_literal(text: Text, literal: str, style: str) -> None:
    start = 0
    while True:
        index = text.plain.find(literal, start)
        if index < 0:
            return
        text.stylize(style, index, index + len(literal))
        start = index + len(literal)


def _style_pattern(text: Text, pattern: re.Pattern[str], style: str) -> None:
    for match in pattern.finditer(text.plain):
        if match.lastindex:
            start, end = match.span(1)
        else:
            start, end = match.span()
        text.stylize(style, start, end)
