import pytest

from ragent_forge.tui.commands import (
    complete_tui_command_suggestion,
    count_tui_command_suggestions,
    format_tui_command_help,
    format_tui_command_suggestions,
    get_tui_command_suggestion_items,
    list_tui_commands,
    match_tui_commands,
    parse_tui_input,
)

EXPECTED_COMMANDS = {
    "ask",
    "search",
    "source",
    "sources",
    "new",
    "sessions",
    "switch",
    "rename",
    "delete",
    "pin",
    "star",
    "session-search",
    "export",
    "branch",
    "rerun",
    "continue-sources",
    "title",
    "turn",
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
        ("/source 2", "source", "2"),
        ("/source next", "source", "next"),
        ("/source prev", "source", "prev"),
        ("/sources", "sources", ""),
        ("/new", "new", ""),
        ("/sessions", "sessions", ""),
        ("/switch session-1", "switch", "session-1"),
        ("/rename Research chat", "rename", "Research chat"),
        ("/delete", "delete", ""),
        ("/pin", "pin", ""),
        ("/star", "star", ""),
        ("/session-search agent", "session-search", "agent"),
        ("/export markdown", "export", "markdown"),
        ("/export json", "export", "json"),
        ("/branch", "branch", ""),
        ("/rerun", "rerun", ""),
        ("/continue-sources", "continue-sources", ""),
        ("/title", "title", ""),
        ("/title auto", "title", "auto"),
        ("/turn next", "turn", "next"),
        ("/docs", "docs", ""),
        ("/trace", "trace", ""),
        ("/settings", "settings", ""),
        ("/config", "settings", ""),
        ("/mode bm25", "mode", "bm25"),
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
        ("/source", "source", "/source <rank|next|prev>"),
        ("/switch", "switch", "/switch <session-id>"),
        ("/rename", "rename", "/rename <title>"),
        ("/session-search", "session-search", "/session-search <query>"),
        ("/export", "export", "/export markdown|json"),
        ("/turn", "turn", "/turn <id|number|next|prev|first|last>"),
        ("/mode", "mode", "/mode lexical|bm25|semantic|hybrid"),
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

    assert {command.name for command in matches} == {
        "search",
        "settings",
        "sessions",
        "session-search",
    }


def test_match_tui_commands_prefix_matches_source_commands() -> None:
    matches = match_tui_commands("/so")

    assert [command.name for command in matches] == ["source", "sources"]


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
        "/source <rank|next|prev>",
        "/sources",
        "/new",
        "/sessions",
        "/switch <session-id>",
        "/rename <title>",
        "/delete",
        "/pin",
        "/star",
        "/session-search <query>",
        "/export markdown|json",
        "/branch",
        "/rerun",
        "/continue-sources",
        "/title [auto|text]",
        "/turn <id|number|next|prev|first|last>",
        "/docs",
        "/trace",
        "/settings",
        "/mode lexical|bm25|semantic|hybrid",
        "/limit <n>",
        "/context <n>",
        "/prompt on|off",
        "/clear",
        "/exit",
    ):
        assert usage in text


@pytest.mark.parametrize("text", ["", "   "])
def test_format_tui_command_suggestions_empty_input_returns_empty(
    text: str,
) -> None:
    assert format_tui_command_suggestions(text) == ""


def test_format_tui_command_suggestions_normal_text_returns_empty() -> None:
    assert format_tui_command_suggestions("What is Agentic RAG?") == ""


def test_format_tui_command_suggestions_slash_returns_limited_suggestions() -> None:
    text = format_tui_command_suggestions("/")

    assert text.startswith("Suggestions:\n")
    assert "/ask <question>" in text
    assert "/search <query>" in text
    assert "/settings" in text
    assert "/mode lexical|bm25|semantic|hybrid" in text
    assert "/limit <n>" not in text
    assert _suggestion_command_count(text) == 8
    assert "use Up/Down for more" in text


def test_format_tui_command_suggestions_scrolls_to_selected_item() -> None:
    text = format_tui_command_suggestions("/", selected_index=8)

    assert "/ask <question>" not in text
    assert "/search <query>" in text
    assert "/sessions" in text
    assert "/new" in text
    assert "/mode lexical|bm25|semantic|hybrid" in text
    assert "> /new" in text
    assert _suggestion_command_count(text) == 8
    assert "use Up/Down for more" in text


def test_format_tui_command_suggestions_prefix_matches_names() -> None:
    text = format_tui_command_suggestions("/se")

    assert "/search <query>" in text
    assert "Search chunks." in text
    assert "/settings" in text
    assert "Show read-only config." in text
    assert "/trace" not in text


def test_format_tui_command_suggestions_prefix_matches_source_commands() -> None:
    text = format_tui_command_suggestions("/so")

    assert "/source <rank|next|prev>" in text
    assert "Select a source by rank, next, or prev." in text
    assert "/sources" in text
    assert "Show current sources." in text


def test_format_tui_command_suggestions_prefix_matches_trace() -> None:
    text = format_tui_command_suggestions("/tr")

    assert text == "Suggestions:\n  /trace  Show latest trace."


def test_format_tui_command_suggestions_alias_uses_canonical_usage() -> None:
    text = format_tui_command_suggestions("/config")

    assert "/settings" in text
    assert "Show read-only config." in text
    assert "/config" not in text


def test_format_tui_command_suggestions_q_alias_uses_canonical_exit_usage() -> None:
    text = format_tui_command_suggestions("/q")

    assert text == "Suggestions:\n  /exit  Exit the TUI."


@pytest.mark.parametrize(
    "text",
    [
        "/search rag",
        "/ask what is RAG",
    ],
)
def test_format_tui_command_suggestions_command_with_args_returns_empty(
    text: str,
) -> None:
    assert format_tui_command_suggestions(text) == ""


def test_format_tui_command_suggestions_unknown_prefix_is_friendly() -> None:
    assert format_tui_command_suggestions("/unknown") == (
        "No matching commands. Type /help for the command list."
    )


def test_format_tui_command_suggestions_limit_is_respected() -> None:
    text = format_tui_command_suggestions("/", limit=2)

    assert _suggestion_command_count(text) == 2
    assert "use Up/Down for more" in text


def test_format_tui_command_suggestions_is_deterministic() -> None:
    first = format_tui_command_suggestions("/s")
    second = format_tui_command_suggestions("/s")

    assert first == second


def test_format_tui_command_suggestions_include_usage_and_description() -> None:
    text = format_tui_command_suggestions("/mode")

    assert text == (
        "Suggestions:\n"
        "  /mode lexical|bm25|semantic|hybrid  "
        "Set retrieval mode: lexical, BM25, semantic, hybrid."
    )


def test_format_tui_command_suggestions_mode_argument_options() -> None:
    text = format_tui_command_suggestions("/mode ", selected_index=0)

    assert text == (
        "Suggestions:\n"
        "> lexical   Use lexical retrieval.\n"
        "  bm25      Use BM25 retrieval.\n"
        "  semantic  Use semantic retrieval.\n"
        "  hybrid    Use hybrid retrieval."
    )


def test_format_tui_command_suggestions_mode_argument_prefix() -> None:
    text = format_tui_command_suggestions("/mode b", selected_index=0)

    assert text == "Suggestions:\n> bm25  Use BM25 retrieval."


def test_format_tui_command_suggestions_prompt_argument_options() -> None:
    text = format_tui_command_suggestions("/prompt o", selected_index=1)

    assert text == (
        "Suggestions:\n"
        "  on   Enable prompt preview.\n"
        "> off  Disable prompt preview."
    )


def test_format_tui_command_suggestions_source_argument_options() -> None:
    text = format_tui_command_suggestions("/source ", selected_index=0)

    assert text == (
        "Suggestions:\n"
        "> next  Inspect next source.\n"
        "  prev  Inspect previous source."
    )


def test_format_tui_command_suggestions_export_argument_options() -> None:
    text = format_tui_command_suggestions("/export ", selected_index=1)

    assert text == (
        "Suggestions:\n"
        "  markdown  Export current session as Markdown.\n"
        "> json      Export current session as JSON."
    )


def test_format_tui_command_suggestions_title_argument_options() -> None:
    text = format_tui_command_suggestions("/title ", selected_index=0)

    assert text == "Suggestions:\n> auto  Generate a short title."


def test_format_tui_command_suggestions_turn_argument_options() -> None:
    text = format_tui_command_suggestions("/turn ", selected_index=2)

    assert text == (
        "Suggestions:\n"
        "  next   Select next answer.\n"
        "  prev   Select previous answer.\n"
        "> first  Select first answer.\n"
        "  last   Select latest answer."
    )


def test_format_tui_command_suggestions_hides_exact_argument_match() -> None:
    assert format_tui_command_suggestions("/mode bm25") == ""
    assert format_tui_command_suggestions("/prompt on") == ""
    assert format_tui_command_suggestions("/source next") == ""
    assert format_tui_command_suggestions("/export markdown") == ""
    assert format_tui_command_suggestions("/title auto") == ""
    assert format_tui_command_suggestions("/turn next") == ""


def test_format_tui_command_suggestions_does_not_mutate_command_registry() -> None:
    before = list_tui_commands()

    _ = format_tui_command_suggestions("/se")

    assert list_tui_commands() == before


def test_get_tui_command_suggestion_items_returns_matching_specs() -> None:
    items = get_tui_command_suggestion_items("/se")

    assert [item.name for item in items] == [
        "search",
        "settings",
        "sessions",
        "session-search",
    ]


def test_get_tui_command_suggestion_items_hides_for_command_arguments() -> None:
    assert get_tui_command_suggestion_items("/search rag") == []


def test_format_tui_command_suggestions_marks_selected_item() -> None:
    text = format_tui_command_suggestions("/se", selected_index=1)

    assert "  /search <query>" in text
    assert "> /settings" in text


def test_complete_tui_command_suggestion_uses_selected_canonical_command() -> None:
    assert complete_tui_command_suggestion("/se", selected_index=1) == "/settings "


def test_complete_tui_command_suggestion_uses_canonical_alias_target() -> None:
    assert complete_tui_command_suggestion("/config", selected_index=0) == "/settings "
    assert complete_tui_command_suggestion("/q", selected_index=0) == "/exit "


def test_complete_tui_command_suggestion_completes_command_arguments() -> None:
    assert complete_tui_command_suggestion("/mode b") == "/mode bm25"
    assert complete_tui_command_suggestion("/prompt o", selected_index=1) == (
        "/prompt off"
    )
    assert complete_tui_command_suggestion("/source n") == "/source next"
    assert complete_tui_command_suggestion("/export j") == "/export json"
    assert complete_tui_command_suggestion("/title a") == "/title auto"
    assert complete_tui_command_suggestion("/turn l") == "/turn last"


def test_count_tui_command_suggestions_counts_arguments() -> None:
    assert count_tui_command_suggestions("/mode ") == 4
    assert count_tui_command_suggestions("/prompt o") == 2
    assert count_tui_command_suggestions("/export ") == 2
    assert count_tui_command_suggestions("/title ") == 1
    assert count_tui_command_suggestions("/turn ") == 4
    assert count_tui_command_suggestions("/mode bm25") == 0


def test_complete_tui_command_suggestion_returns_none_without_candidates() -> None:
    assert complete_tui_command_suggestion("hello", selected_index=0) is None
    assert complete_tui_command_suggestion("/search rag", selected_index=0) is None


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


def _suggestion_command_count(text: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if line.startswith("  /") or line.startswith("> /")
    )
