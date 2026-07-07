from rich.text import Text

from ragent_forge.tui.theme import (
    style_command_suggestions,
    style_inspector,
    style_shell_status,
    style_transcript,
)


def test_shell_status_styles_mode_prompt_and_running_status() -> None:
    plain = "mode: hybrid | limit: 5 | context: 4000 | prompt: off | status: running"

    styled = style_shell_status(plain)

    assert styled.plain == plain
    assert _has_style_covering(styled, "hybrid", "bold green")
    assert _has_style_covering(styled, "off", "dim")
    assert _has_style_covering(styled, "running", "bold yellow")


def test_transcript_styles_roles_sources_and_error_heading() -> None:
    plain = "\n".join(
        [
            "Assistant:",
            "  Agentic RAG adds planning.",
            "",
            "Sources:",
            "1. agentic_rag.md  score=1  chunk=chunk-0001",
            "",
            "Error:",
            "  Generation failed.",
        ]
    )

    styled = style_transcript(plain)

    assert styled.plain == plain
    assert _has_style_covering(styled, "Assistant:", "bold green")
    assert _has_style_covering(styled, "Sources:", "bold cyan")
    assert _has_style_covering(styled, "score=1", "yellow")
    assert _has_style_covering(styled, "Error:", "bold red")


def test_inspector_styles_section_headings_keys_and_retrieval_modes() -> None:
    plain = "\n".join(
        [
            "Shell details",
            "",
            "mode: bm25",
            "status: ready",
            "",
            "Retrieval metadata",
            "",
            "method: hybrid_rrf",
        ]
    )

    styled = style_inspector(plain)

    assert styled.plain == plain
    assert _has_style_covering(styled, "Shell details", "bold cyan")
    assert _has_style_covering(styled, "mode:", "dim")
    assert _has_style_covering(styled, "bm25", "bright_magenta")
    assert _has_style_covering(styled, "ready", "green")
    assert _has_style_covering(styled, "hybrid_rrf", "bold green")


def test_command_suggestions_styles_selected_command_and_usage() -> None:
    plain = "\n".join(
        [
            "Suggestions:",
            "  /search <query>  Search chunks.",
            "> /settings        Show read-only config.",
        ]
    )

    styled = style_command_suggestions(plain)

    assert styled.plain == plain
    assert _has_style_covering(styled, "Suggestions:", "bold cyan")
    assert _has_style_covering(styled, "> /settings", "bold reverse")
    assert _has_style_covering(styled, "/search", "cyan")


def _has_style_covering(text: Text, substring: str, style: str) -> bool:
    start = text.plain.index(substring)
    end = start + len(substring)
    return any(
        span.start <= start and span.end >= end and str(span.style) == style
        for span in text.spans
    )
