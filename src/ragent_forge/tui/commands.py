from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TuiCommandName = Literal[
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
        return sorted(commands, key=lambda command: command.name)

    matches = [
        command
        for command in commands
        if command.name.startswith(normalized)
        or any(alias.startswith(normalized) for alias in command.aliases)
    ]
    return sorted(matches, key=lambda command: command.name)


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
