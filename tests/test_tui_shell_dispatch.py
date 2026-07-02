from ragent_forge.tui.commands import format_tui_command_help
from ragent_forge.tui.shell_dispatch import apply_shell_input
from ragent_forge.tui.shell_models import (
    TranscriptMessage,
    create_initial_shell_state,
    format_shell_status,
    format_transcript,
)

ASK_NOT_WIRED = "Ask execution from Shell is not wired yet. Use the Ask page for now."


def test_shell_initial_state_renders_status_and_welcome_transcript() -> None:
    state = create_initial_shell_state()

    assert format_shell_status(state) == (
        "mode: lexical | limit: 5 | context: 4000 | prompt: off | status: idle"
    )
    assert "RAGentForge command shell." in format_transcript(state.messages)


def test_apply_shell_input_normal_text_appends_ask_placeholder() -> None:
    result = apply_shell_input(create_initial_shell_state(), "What is Agentic RAG?")

    assert result.action == "none"
    assert result.state.messages[-2:] == (
        TranscriptMessage(role="user", text="What is Agentic RAG?"),
        TranscriptMessage(role="tool", text=ASK_NOT_WIRED),
    )


def test_apply_shell_input_ask_command_appends_ask_placeholder() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/ask What is RAG?")

    assert result.action == "none"
    assert result.state.messages[-2:] == (
        TranscriptMessage(role="user", text="What is RAG?"),
        TranscriptMessage(role="tool", text=ASK_NOT_WIRED),
    )


def test_apply_shell_input_help_appends_command_help() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/help")

    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text=format_tui_command_help(),
    )


def test_apply_shell_input_clear_resets_transcript() -> None:
    state = apply_shell_input(create_initial_shell_state(), "question").state

    result = apply_shell_input(state, "/clear")

    assert result.state.messages == create_initial_shell_state().messages


def test_apply_shell_input_mode_updates_retrieval_mode() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/mode hybrid")

    assert result.state.retrieval_mode == "hybrid"
    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="retrieval mode set to hybrid",
    )


def test_apply_shell_input_invalid_mode_appends_error_without_changing_mode() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/mode bm25")

    assert result.state.retrieval_mode == "lexical"
    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Invalid retrieval mode: bm25",
    )


def test_apply_shell_input_limit_updates_limit() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/limit 7")

    assert result.state.limit == 7
    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="limit set to 7",
    )


def test_apply_shell_input_non_positive_limit_appends_error_without_change() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/limit 0")

    assert result.state.limit == 5
    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Limit must be positive.",
    )


def test_apply_shell_input_invalid_limit_appends_error_without_changing_limit() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/limit nope")

    assert result.state.limit == 5
    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Invalid limit: nope",
    )


def test_apply_shell_input_context_updates_max_context_chars() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/context 2000")

    assert result.state.max_context_chars == 2000
    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="max context chars set to 2000",
    )


def test_apply_shell_input_invalid_context_appends_error_without_change() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/context nope")

    assert result.state.max_context_chars == 4000
    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Invalid context value: nope",
    )


def test_apply_shell_input_prompt_on_and_off_toggle_preview() -> None:
    enabled = apply_shell_input(create_initial_shell_state(), "/prompt on")
    disabled = apply_shell_input(enabled.state, "/prompt off")

    assert enabled.state.show_prompt is True
    assert enabled.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="prompt preview enabled",
    )
    assert disabled.state.show_prompt is False
    assert disabled.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="prompt preview disabled",
    )


def test_apply_shell_input_invalid_prompt_appends_usage_error() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/prompt maybe")

    assert result.state.show_prompt is False
    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Usage: /prompt on|off",
    )


def test_apply_shell_input_planned_not_wired_commands_append_messages() -> None:
    expectations = {
        "/search agent memory": (
            "/search dispatch is not wired yet. Use the Search page for now."
        ),
        "/docs": "/docs dispatch is not wired yet. Use the Documents page for now.",
        "/trace": "/trace dispatch is not wired yet. Use the Trace page for now.",
        "/settings": (
            "/settings dispatch is not wired yet. Use the Settings page for now."
        ),
    }

    for command, message in expectations.items():
        result = apply_shell_input(create_initial_shell_state(), command)
        assert result.state.messages[-1] == TranscriptMessage(
            role="tool",
            text=message,
        )


def test_apply_shell_input_unknown_command_appends_error() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/unknown value")

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Unknown command: /unknown",
    )


def test_apply_shell_input_missing_arguments_appends_parser_error() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/search")

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Missing arguments for /search. Usage: /search <query>",
    )


def test_apply_shell_input_empty_input_appends_parser_error() -> None:
    result = apply_shell_input(create_initial_shell_state(), "   ")

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Enter a question or slash command.",
    )


def test_apply_shell_input_exit_returns_quit_action() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/q")

    assert result.action == "quit"


def test_shell_transcript_formatting_hides_api_keys_from_shell_messages() -> None:
    result = apply_shell_input(create_initial_shell_state(), "api_key=abc123")

    rendered = format_transcript(result.state.messages)

    assert "abc123" not in rendered
    assert "<hidden>" in rendered
