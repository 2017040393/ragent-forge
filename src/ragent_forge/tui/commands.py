from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TuiCommandName = Literal[
    "ask",
    "search",
    "source",
    "sources",
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


@dataclass(frozen=True)
class SlashCommandSpec:
    name: str
    aliases: tuple[str, ...]
    description: str
    usage: str
    requires_args: bool = False


@dataclass(frozen=True)
class ParsedTuiCommand:
    name: TuiCommandName
    args: str
    raw: str
    is_slash_command: bool
    error: str | None = None


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
            name="settings",
            aliases=("config",),
            description="Show read-only config.",
            usage="/settings",
        ),
        SlashCommandSpec(
            name="mode",
            aliases=(),
            description="Set retrieval mode: lexical, semantic, hybrid.",
            usage="/mode lexical|semantic|hybrid",
            requires_args=True,
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
            name="help",
            aliases=(),
            description="Show available commands.",
            usage="/help",
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


def get_tui_command_suggestion_items(
    text: str,
    *,
    limit: int = 6,
) -> list[SlashCommandSpec]:
    if limit <= 0:
        return []

    raw = text.lstrip()
    if not raw or not raw.startswith("/"):
        return []

    command_fragment = raw[1:]
    if any(character.isspace() for character in command_fragment):
        return []

    return match_tui_commands(command_fragment)[:limit]


def complete_tui_command_suggestion(
    text: str,
    *,
    selected_index: int = 0,
    limit: int = 6,
) -> str | None:
    items = get_tui_command_suggestion_items(text, limit=limit)
    if not items:
        return None

    selected = items[selected_index % len(items)]
    return f"/{selected.name} "


def format_tui_command_suggestions(
    text: str,
    *,
    limit: int = 6,
    selected_index: int | None = None,
) -> str:
    visible_matches = get_tui_command_suggestion_items(text, limit=limit)
    if not visible_matches:
        raw = text.lstrip()
        if not raw or not raw.startswith("/"):
            return ""
        command_fragment = raw[1:]
        if not command_fragment or any(
            character.isspace() for character in command_fragment
        ):
            return ""
        return "No matching commands. Type /help for the command list."

    usage_width = max(len(command.usage) for command in visible_matches)
    lines = ["Suggestions:"]
    selected_match_index = (
        selected_index % len(visible_matches)
        if selected_index is not None
        else None
    )
    for index, command in enumerate(visible_matches):
        marker = ">" if index == selected_match_index else " "
        lines.append(
            f"{marker} {command.usage.ljust(usage_width)}  {command.description}"
        )

    if len(match_tui_commands(text.lstrip()[1:])) > limit:
        lines.append("  ... type more to narrow results")
    return "\n".join(lines)


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
