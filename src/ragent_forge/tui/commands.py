from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TuiCommandName = Literal[
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
    "unknown",
]

DEFAULT_SUGGESTION_LIMIT = 8


@dataclass(frozen=True)
class SlashCommandSpec:
    name: str
    aliases: tuple[str, ...]
    description: str
    usage: str
    requires_args: bool = False


@dataclass(frozen=True)
class SlashArgumentSuggestion:
    value: str
    description: str


@dataclass(frozen=True)
class ParsedTuiCommand:
    name: TuiCommandName
    args: str
    raw: str
    is_slash_command: bool
    error: str | None = None


@dataclass(frozen=True)
class _ArgumentSuggestionContext:
    command: SlashCommandSpec
    matches: list[SlashArgumentSuggestion]


_ARGUMENT_SUGGESTIONS = {
    "mode": (
        SlashArgumentSuggestion("lexical", "Use lexical retrieval."),
        SlashArgumentSuggestion("bm25", "Use BM25 retrieval."),
        SlashArgumentSuggestion("semantic", "Use semantic retrieval."),
        SlashArgumentSuggestion("hybrid", "Use hybrid retrieval."),
    ),
    "prompt": (
        SlashArgumentSuggestion("on", "Enable prompt preview."),
        SlashArgumentSuggestion("off", "Disable prompt preview."),
    ),
    "source": (
        SlashArgumentSuggestion("next", "Inspect next source."),
        SlashArgumentSuggestion("prev", "Inspect previous source."),
    ),
    "export": (
        SlashArgumentSuggestion("markdown", "Export current session as Markdown."),
        SlashArgumentSuggestion("json", "Export current session as JSON."),
    ),
    "title": (
        SlashArgumentSuggestion("auto", "Generate a short title."),
    ),
    "turn": (
        SlashArgumentSuggestion("next", "Select next answer."),
        SlashArgumentSuggestion("prev", "Select previous answer."),
        SlashArgumentSuggestion("first", "Select first answer."),
        SlashArgumentSuggestion("last", "Select latest answer."),
    ),
}


def list_tui_commands() -> list[SlashCommandSpec]:
    return [
        SlashCommandSpec(
            name="ask",
            aliases=(),
            description="Ask a question.",
            usage="/ask <question>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="search",
            aliases=("s",),
            description="Search chunks.",
            usage="/search <query>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="source",
            aliases=(),
            description="Select a source by rank, next, or prev.",
            usage="/source <rank|next|prev>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="sources",
            aliases=(),
            description="Show current sources.",
            usage="/sources",
        ),
        SlashCommandSpec(
            name="settings",
            aliases=("config",),
            description="Show read-only config.",
            usage="/settings",
        ),
        SlashCommandSpec(
            name="mode",
            aliases=(),
            description="Set retrieval mode: lexical, BM25, semantic, hybrid.",
            usage="/mode lexical|bm25|semantic|hybrid",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="help",
            aliases=(),
            description="Show available commands.",
            usage="/help",
        ),
        SlashCommandSpec(
            name="sessions",
            aliases=(),
            description="Show saved sessions.",
            usage="/sessions",
        ),
        SlashCommandSpec(
            name="new",
            aliases=(),
            description="Start a new session.",
            usage="/new",
        ),
        SlashCommandSpec(
            name="switch",
            aliases=(),
            description="Switch to a session.",
            usage="/switch <session-id>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="rename",
            aliases=(),
            description="Rename current session.",
            usage="/rename <title>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="delete",
            aliases=(),
            description="Delete current session.",
            usage="/delete",
        ),
        SlashCommandSpec(
            name="pin",
            aliases=(),
            description="Toggle current session pin.",
            usage="/pin",
        ),
        SlashCommandSpec(
            name="star",
            aliases=(),
            description="Toggle current session star.",
            usage="/star",
        ),
        SlashCommandSpec(
            name="session-search",
            aliases=(),
            description="Search saved sessions.",
            usage="/session-search <query>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="export",
            aliases=(),
            description="Export current session.",
            usage="/export markdown|json",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="branch",
            aliases=(),
            description="Branch from selected answer.",
            usage="/branch",
        ),
        SlashCommandSpec(
            name="rerun",
            aliases=(),
            description="Rerun selected question.",
            usage="/rerun",
        ),
        SlashCommandSpec(
            name="continue-sources",
            aliases=(),
            description="Draft a follow-up using selected sources.",
            usage="/continue-sources",
        ),
        SlashCommandSpec(
            name="title",
            aliases=(),
            description="Show, set, or auto-generate the session title.",
            usage="/title [auto|text]",
        ),
        SlashCommandSpec(
            name="turn",
            aliases=(),
            description="Select an answer turn.",
            usage="/turn <id|number|next|prev|first|last>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="docs",
            aliases=(),
            description="Show document summary.",
            usage="/docs",
        ),
        SlashCommandSpec(
            name="trace",
            aliases=("t",),
            description="Show latest trace.",
            usage="/trace",
        ),
        SlashCommandSpec(
            name="limit",
            aliases=(),
            description="Set retrieval result limit.",
            usage="/limit <n>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="context",
            aliases=(),
            description="Set max context chars.",
            usage="/context <n>",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="prompt",
            aliases=(),
            description="Toggle prompt preview.",
            usage="/prompt on|off",
            requires_args=True,
        ),
        SlashCommandSpec(
            name="clear",
            aliases=(),
            description="Clear transcript.",
            usage="/clear",
        ),
        SlashCommandSpec(
            name="exit",
            aliases=("quit", "q"),
            description="Exit the TUI.",
            usage="/exit",
        ),
    ]


def parse_tui_input(text: str) -> ParsedTuiCommand:
    stripped = text.strip()
    if not stripped:
        return ParsedTuiCommand(
            name="unknown",
            args="",
            raw=text,
            is_slash_command=False,
            error="Enter a question or slash command.",
        )

    if not stripped.startswith("/"):
        return ParsedTuiCommand(
            name="ask",
            args=stripped,
            raw=text,
            is_slash_command=False,
        )

    command_text = stripped[1:].strip()
    if not command_text:
        return ParsedTuiCommand(
            name="unknown",
            args="",
            raw=text,
            is_slash_command=True,
            error="Unknown command: /",
        )

    command_token, args = _split_command(command_text)
    normalized_token = command_token.lower()
    spec = _command_spec_by_token().get(normalized_token)
    if spec is None:
        return ParsedTuiCommand(
            name="unknown",
            args=args,
            raw=text,
            is_slash_command=True,
            error=f"Unknown command: /{command_token}",
        )

    if spec.requires_args and not args:
        return ParsedTuiCommand(
            name=_typed_command_name(spec.name),
            args="",
            raw=text,
            is_slash_command=True,
            error=(
                f"Missing arguments for /{spec.name}. "
                f"Usage: {spec.usage}"
            ),
        )

    return ParsedTuiCommand(
        name=_typed_command_name(spec.name),
        args=args,
        raw=text,
        is_slash_command=True,
    )


def match_tui_commands(prefix: str) -> list[SlashCommandSpec]:
    normalized = prefix.strip()
    if normalized.startswith("/"):
        normalized = normalized[1:]
    normalized = normalized.lower()

    commands = list_tui_commands()
    if not normalized:
        return commands

    return [
        command
        for command in commands
        if command.name.startswith(normalized)
        or any(alias.startswith(normalized) for alias in command.aliases)
    ]


def format_tui_command_help(
    commands: list[SlashCommandSpec] | None = None,
) -> str:
    command_specs = commands or list_tui_commands()
    usage_width = max(len(command.usage) for command in command_specs)
    lines = ["Slash commands", ""]
    lines.extend(
        f"{command.usage.ljust(usage_width)}  {command.description}"
        for command in command_specs
    )
    return "\n".join(lines)


def format_tui_task_help(
    commands: list[SlashCommandSpec] | None = None,
) -> str:
    command_specs = commands or list_tui_commands()
    specs_by_name = {command.name: command for command in command_specs}
    lines = [
        "TUI help",
        "",
        "Type a question to ask. Type / for command suggestions.",
    ]
    for section, command_names, example in _TUI_HELP_GROUPS:
        section_commands = [
            specs_by_name[name] for name in command_names if name in specs_by_name
        ]
        if not section_commands:
            continue
        usage_width = max(len(command.usage) for command in section_commands)
        lines.extend(["", section])
        lines.extend(
            f"{command.usage.ljust(usage_width)}  {command.description}"
            for command in section_commands
        )
        lines.append(f"Example: {example}")
    return "\n".join(lines)


_TUI_HELP_GROUPS = (
    ("Ask", ("ask", "context", "prompt"), "What is Agentic RAG?"),
    ("Search", ("search", "mode", "limit"), "/search Agentic RAG"),
    ("Sources", ("sources", "source", "turn", "continue-sources"), "/source next"),
    (
        "Sessions",
        (
            "sessions",
            "new",
            "switch",
            "rename",
            "delete",
            "pin",
            "star",
            "session-search",
            "export",
            "branch",
            "rerun",
            "title",
        ),
        "/export markdown",
    ),
    ("Workspace", ("docs", "settings"), "/docs"),
    ("Debug", ("trace",), "/trace"),
    ("General", ("help", "clear", "exit"), "/help"),
)


def get_tui_command_suggestion_items(
    text: str,
    *,
    limit: int = DEFAULT_SUGGESTION_LIMIT,
    selected_index: int | None = None,
) -> list[SlashCommandSpec]:
    if limit <= 0:
        return []

    matches = get_tui_command_suggestion_matches(text)
    if not matches:
        return []

    start = _suggestion_window_start(
        total_count=len(matches),
        limit=limit,
        selected_index=selected_index,
    )
    return matches[start : start + limit]


def get_tui_command_suggestion_matches(text: str) -> list[SlashCommandSpec]:
    raw = text.lstrip()
    if not raw or not raw.startswith("/"):
        return []

    command_fragment = raw[1:]
    if any(character.isspace() for character in command_fragment):
        return []

    return match_tui_commands(command_fragment)


def count_tui_command_suggestions(text: str) -> int:
    argument_context = _argument_suggestion_context(text)
    if argument_context is not None:
        return len(argument_context.matches)
    return len(get_tui_command_suggestion_matches(text))


def _suggestion_window_start(
    *,
    total_count: int,
    limit: int,
    selected_index: int | None,
) -> int:
    visible_count = min(limit, total_count)
    if selected_index is None or total_count <= visible_count:
        return 0

    selected_match_index = selected_index % total_count
    return min(
        max(0, selected_match_index - visible_count + 1),
        total_count - visible_count,
    )


def _is_slash_command_prefix(text: str) -> bool:
    raw = text.lstrip()
    if not raw or not raw.startswith("/"):
        return False

    command_fragment = raw[1:]
    return bool(command_fragment) and not any(
        character.isspace() for character in command_fragment
    )


def complete_tui_command_suggestion(
    text: str,
    *,
    selected_index: int = 0,
    limit: int = DEFAULT_SUGGESTION_LIMIT,
) -> str | None:
    if limit <= 0:
        return None

    argument_context = _argument_suggestion_context(text)
    if argument_context is not None:
        if not argument_context.matches:
            return None
        selected = argument_context.matches[
            selected_index % len(argument_context.matches)
        ]
        return f"/{argument_context.command.name} {selected.value}"

    items = get_tui_command_suggestion_matches(text)
    if not items:
        return None

    selected = items[selected_index % len(items)]
    return f"/{selected.name} "


def format_tui_command_suggestions(
    text: str,
    *,
    limit: int = DEFAULT_SUGGESTION_LIMIT,
    selected_index: int | None = None,
) -> str:
    argument_context = _argument_suggestion_context(text)
    if argument_context is not None:
        return _format_argument_suggestions(argument_context, selected_index)

    all_matches = get_tui_command_suggestion_matches(text)
    if not all_matches:
        if not _is_slash_command_prefix(text):
            return ""
        return "No matching commands. Type /help for the command list."

    visible_matches = get_tui_command_suggestion_items(
        text,
        limit=limit,
        selected_index=selected_index,
    )
    usage_width = max(len(command.usage) for command in visible_matches)
    lines = ["Suggestions:"]
    selected_match_index = (
        selected_index % len(all_matches)
        if selected_index is not None
        else None
    )
    window_start = _suggestion_window_start(
        total_count=len(all_matches),
        limit=limit,
        selected_index=selected_index,
    )
    for index, command in enumerate(visible_matches, start=window_start):
        marker = ">" if index == selected_match_index else " "
        lines.append(
            f"{marker} {command.usage.ljust(usage_width)}  {command.description}"
        )

    if len(all_matches) > limit:
        lines.append("  ... use Up/Down for more")
    return "\n".join(lines)


def _format_argument_suggestions(
    context: _ArgumentSuggestionContext,
    selected_index: int | None,
) -> str:
    if not context.matches:
        return ""

    value_width = max(len(item.value) for item in context.matches)
    selected_match_index = (
        selected_index % len(context.matches) if selected_index is not None else None
    )
    lines = ["Suggestions:"]
    for index, item in enumerate(context.matches):
        marker = ">" if index == selected_match_index else " "
        lines.append(
            f"{marker} {item.value.ljust(value_width)}  {item.description}"
        )
    return "\n".join(lines)


def _argument_suggestion_context(text: str) -> _ArgumentSuggestionContext | None:
    raw = text.lstrip()
    if not raw or not raw.startswith("/"):
        return None

    command_fragment = raw[1:]
    if not any(character.isspace() for character in command_fragment):
        return None

    command_token, args = _split_command(command_fragment)
    spec = _command_spec_by_token().get(command_token.lower())
    if spec is None:
        return None

    suggestions = _ARGUMENT_SUGGESTIONS.get(spec.name)
    if suggestions is None:
        return None

    normalized_args = args.lower()
    if " " in normalized_args:
        return _ArgumentSuggestionContext(spec, [])

    matches = [
        suggestion
        for suggestion in suggestions
        if suggestion.value.startswith(normalized_args)
    ]
    if (
        len(matches) == 1
        and normalized_args
        and matches[0].value == normalized_args
    ):
        matches = []
    return _ArgumentSuggestionContext(spec, matches)


def _split_command(command_text: str) -> tuple[str, str]:
    parts = command_text.split(maxsplit=1)
    command = parts[0]
    args = parts[1].strip() if len(parts) > 1 else ""
    return command, args


def _command_spec_by_token() -> dict[str, SlashCommandSpec]:
    specs: dict[str, SlashCommandSpec] = {}
    for command in list_tui_commands():
        specs[command.name] = command
        for alias in command.aliases:
            specs[alias] = command
    return specs


def _typed_command_name(name: str) -> TuiCommandName:
    if name in {
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
    }:
        return name  # type: ignore[return-value]
    return "unknown"
