import pytest

from ragent_forge.tui.commands import (
    format_tui_command_help,
    list_tui_commands,
    match_tui_commands,
    parse_tui_input,
)

EXPECTED_COMMANDS = {
    "ask",
    "search",
    "docs",
    "trace",
    "settings",
    "mode",
    "limit",
    "context",
    "prompt",
    "help",
    "clear",
    "exit",
}


def test_list_tui_commands_includes_expected_commands() -> None:
    names = {command.name for command in list_tui_commands()}

    assert names == EXPECTED_COMMANDS


def test_command_names_are_unique() -> None:
    names = [command.name for command in list_tui_commands()]

    assert len(names) == len(set(names))


def test_command_aliases_do_not_conflict() -> None:
    commands = list_tui_commands()
    names = {command.name for command in commands}
    aliases = [
        alias
        for command in commands
        for alias in command.aliases
    ]

    assert len(aliases) == len(set(aliases))
    assert set(aliases).isdisjoint(names)


def test_empty_input_returns_unknown_with_helpful_error() -> None:
    parsed = parse_tui_input("   ")

    assert parsed.name == "unknown"
    assert parsed.args == ""
    assert parsed.raw == "   "
    assert parsed.is_slash_command is False
    assert parsed.error == "Enter a question or slash command."


def test_normal_text_parses_as_ask() -> None:
    parsed = parse_tui_input("   What is Agentic RAG?   ")

    assert parsed.name == "ask"
    assert parsed.args == "What is Agentic RAG?"
    assert parsed.raw == "   What is Agentic RAG?   "
    assert parsed.is_slash_command is False
    assert parsed.error is None


@pytest.mark.parametrize(
    ("text", "expected_name", "expected_args"),
    [
        ("/ask What is RAG?", "ask", "What is RAG?"),
        ("/search hybrid retrieval", "search", "hybrid retrieval"),
        ("/docs", "docs", ""),
        ("/trace", "trace", ""),
        ("/settings", "settings", ""),
        ("/config", "settings", ""),
        ("/mode hybrid", "mode", "hybrid"),
        ("/limit 5", "limit", "5"),
        ("/context 4000", "context", "4000"),
        ("/prompt on", "prompt", "on"),
        ("/clear", "clear", ""),
        ("/exit", "exit", ""),
        ("/quit", "exit", ""),
        ("/q", "exit", ""),
    ],
)
def test_known_slash_commands_parse(
    text: str,
    expected_name: str,
    expected_args: str,
) -> None:
    parsed = parse_tui_input(text)

    assert parsed.name == expected_name
    assert parsed.args == expected_args
    assert parsed.is_slash_command is True
    assert parsed.error is None


def test_unknown_slash_command_returns_unknown_with_error() -> None:
    parsed = parse_tui_input("/unknown value")

    assert parsed.name == "unknown"
    assert parsed.args == "value"
    assert parsed.is_slash_command is True
    assert parsed.error == "Unknown command: /unknown"


@pytest.mark.parametrize(
    ("text", "command", "usage"),
    [
        ("/ask", "ask", "/ask <question>"),
        ("/search", "search", "/search <query>"),
        ("/mode", "mode", "/mode lexical|semantic|hybrid"),
        ("/limit", "limit", "/limit <n>"),
        ("/context", "context", "/context <n>"),
        ("/prompt", "prompt", "/prompt on|off"),
    ],
)
def test_missing_required_args_returns_usage_error(
    text: str,
    command: str,
    usage: str,
) -> None:
    parsed = parse_tui_input(text)

    assert parsed.name == command
    assert parsed.args == ""
    assert parsed.is_slash_command is True
    assert parsed.error == f"Missing arguments for /{command}. Usage: {usage}"


def test_match_tui_commands_slash_returns_all_commands() -> None:
    matches = match_tui_commands("/")

    assert {command.name for command in matches} == EXPECTED_COMMANDS


def test_match_tui_commands_prefix_matches_names_and_aliases() -> None:
    matches = match_tui_commands("/se")

    assert {command.name for command in matches} == {"search", "settings"}


def test_match_tui_commands_without_slash_matches_trace() -> None:
    matches = match_tui_commands("tr")

    assert [command.name for command in matches] == ["trace"]


def test_format_tui_command_help_includes_major_commands() -> None:
    text = format_tui_command_help()

    assert text.startswith("Slash commands")
    for usage in (
        "/help",
        "/ask <question>",
        "/search <query>",
        "/docs",
        "/trace",
        "/settings",
        "/mode lexical|semantic|hybrid",
        "/limit <n>",
        "/context <n>",
        "/prompt on|off",
        "/clear",
        "/exit",
    ):
        assert usage in text


@pytest.mark.parametrize(
    "text",
    [
        "/",
        "////",
        "/ search",
        "   /help   ",
        "   normal question   ",
    ],
)
def test_parser_handles_weird_input_without_raising(text: str) -> None:
    parsed = parse_tui_input(text)

    assert parsed.raw == text
