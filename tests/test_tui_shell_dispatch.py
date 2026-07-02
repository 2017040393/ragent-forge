from ragent_forge.tui.commands import format_tui_command_help
from ragent_forge.tui.shell_dispatch import ShellReadOnlyHandlers, apply_shell_input
from ragent_forge.tui.shell_models import (
    ShellState,
    TranscriptMessage,
    TranscriptSource,
    create_initial_shell_state,
    format_shell_status,
    format_transcript,
)


def make_source(rank: int = 1) -> TranscriptSource:
    return TranscriptSource(
        rank=rank,
        chunk_id=f"/knowledge/rag.md::chunk-{rank:04d}",
        source_path=f"/knowledge/source_{rank}.md",
        score=0.1 * rank,
        preview=f"Preview {rank}",
        metadata={"retrieval_method": "lexical_token_overlap"},
    )


def test_shell_initial_state_renders_status_and_welcome_transcript() -> None:
    state = create_initial_shell_state()

    assert format_shell_status(state) == (
        "mode: lexical | limit: 5 | context: 4000 | prompt: off | status: idle"
    )
    assert "RAGentForge command shell." in format_transcript(state.messages)


def test_apply_shell_input_normal_text_returns_ask_action() -> None:
    result = apply_shell_input(create_initial_shell_state(), "What is Agentic RAG?")

    assert result.action == "ask"
    assert result.ask_question == "What is Agentic RAG?"
    assert result.search_query is None
    assert result.state == create_initial_shell_state()


def test_apply_shell_input_ask_command_returns_ask_action() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/ask What is RAG?")

    assert result.action == "ask"
    assert result.ask_question == "What is RAG?"
    assert result.search_query is None
    assert result.state == create_initial_shell_state()


def test_apply_shell_input_ask_without_question_appends_parser_error() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/ask")

    assert result.action == "none"
    assert result.ask_question is None
    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Missing arguments for /ask. Usage: /ask <question>",
    )


def test_apply_shell_input_search_returns_search_action() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/search agent memory")

    assert result.action == "search"
    assert result.search_query == "agent memory"
    assert result.ask_question is None
    assert result.state == create_initial_shell_state()


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


def test_apply_shell_input_sources_without_sources_appends_friendly_message() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/sources")

    assert result.action == "none"
    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="No sources available. Run /search <query> or ask a question first.",
    )


def test_apply_shell_input_sources_with_sources_appends_source_list_message() -> None:
    first = make_source(1)
    second = make_source(2)
    state = ShellState(available_sources=(first, second), selected_source=first)

    result = apply_shell_input(state, "/sources")

    assert result.action == "none"
    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="Current sources",
        metadata={"operation": "source_list"},
        sources=(first, second),
    )
    assert "Sources:" in format_transcript((result.state.messages[-1],))


def test_apply_shell_input_sources_preserves_selected_source() -> None:
    first = make_source(1)
    second = make_source(2)
    state = ShellState(
        available_sources=(first, second),
        selected_source=second,
    )

    result = apply_shell_input(state, "/sources")

    assert result.action == "none"
    assert result.state.available_sources == (first, second)
    assert result.state.selected_source == second
    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="Current sources",
        metadata={"operation": "source_list"},
        sources=(first, second),
    )


def test_apply_shell_input_source_rank_selects_source() -> None:
    first = make_source(1)
    second = make_source(2)
    state = ShellState(available_sources=(first, second), selected_source=first)

    result = apply_shell_input(state, "/source 2")

    assert result.action == "none"
    assert result.search_query is None
    assert result.ask_question is None
    assert result.state.selected_source == second
    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="selected source 2: source_2.md",
    )


def test_apply_shell_input_source_next_selects_next_and_wraps() -> None:
    first = make_source(1)
    second = make_source(2)
    state = ShellState(available_sources=(first, second), selected_source=first)

    moved = apply_shell_input(state, "/source next")
    wrapped = apply_shell_input(moved.state, "/source next")

    assert moved.state.selected_source == second
    assert moved.state.messages[-1].text == "selected source 2: source_2.md"
    assert wrapped.state.selected_source == first
    assert wrapped.state.messages[-1].text == "selected source 1: source_1.md"


def test_apply_shell_input_source_prev_selects_previous_and_wraps() -> None:
    first = make_source(1)
    second = make_source(2)
    state = ShellState(available_sources=(first, second), selected_source=second)

    moved = apply_shell_input(state, "/source prev")
    wrapped = apply_shell_input(moved.state, "/source prev")

    assert moved.state.selected_source == first
    assert moved.state.messages[-1].text == "selected source 1: source_1.md"
    assert wrapped.state.selected_source == second
    assert wrapped.state.messages[-1].text == "selected source 2: source_2.md"


def test_apply_shell_input_source_without_sources_appends_friendly_message() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/source 1")

    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="No sources available. Run /search <query> or ask a question first.",
    )


def test_apply_shell_input_source_zero_appends_friendly_error() -> None:
    state = ShellState(available_sources=(make_source(1),))

    result = apply_shell_input(state, "/source 0")

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Source rank must be a positive integer.",
    )


def test_apply_shell_input_source_invalid_arg_appends_usage_error() -> None:
    state = ShellState(available_sources=(make_source(1),))

    result = apply_shell_input(state, "/source nope")

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Usage: /source <rank|next|prev>",
    )


def test_apply_shell_input_source_out_of_range_appends_friendly_error() -> None:
    state = ShellState(available_sources=(make_source(1), make_source(2)))

    result = apply_shell_input(state, "/source 99")

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Source rank out of range. Available sources: 1-2.",
    )


def test_apply_shell_input_planned_not_wired_commands_append_messages() -> None:
    expectations = {
        "/docs": "/docs is unavailable in this shell context.",
        "/trace": "/trace is unavailable in this shell context.",
        "/settings": "/settings is unavailable in this shell context.",
    }

    for command, message in expectations.items():
        result = apply_shell_input(create_initial_shell_state(), command)
        assert result.state.messages[-1] == TranscriptMessage(
            role="tool",
            text=message,
        )


def test_apply_shell_input_config_without_handlers_uses_settings_fallback() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/config")

    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="/settings is unavailable in this shell context.",
    )


def test_apply_shell_input_docs_handler_appends_tool_output() -> None:
    result = apply_shell_input(
        create_initial_shell_state(),
        "/docs",
        handlers=ShellReadOnlyHandlers(docs=lambda: "Workspace\n  status: ready"),
    )

    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="Workspace\n  status: ready",
        metadata={"operation": "shell_command", "command": "docs"},
    )


def test_apply_shell_input_trace_handler_appends_tool_output() -> None:
    result = apply_shell_input(
        create_initial_shell_state(),
        "/trace",
        handlers=ShellReadOnlyHandlers(trace=lambda: "Latest trace\n\nSteps"),
    )

    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="Latest trace\n\nSteps",
        metadata={"operation": "shell_command", "command": "trace"},
    )


def test_apply_shell_input_settings_handler_appends_tool_output() -> None:
    result = apply_shell_input(
        create_initial_shell_state(),
        "/settings",
        handlers=ShellReadOnlyHandlers(settings=lambda: "config path: .ragent"),
    )

    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="config path: .ragent",
        metadata={"operation": "shell_command", "command": "settings"},
    )


def test_apply_shell_input_config_uses_settings_handler() -> None:
    result = apply_shell_input(
        create_initial_shell_state(),
        "/config",
        handlers=ShellReadOnlyHandlers(settings=lambda: "config path: .ragent"),
    )

    assert result.state.messages[-1] == TranscriptMessage(
        role="tool",
        text="config path: .ragent",
        metadata={"operation": "shell_command", "command": "settings"},
    )


def test_apply_shell_input_docs_handler_error_appends_friendly_error() -> None:
    def fail() -> str:
        raise RuntimeError("broken docs")

    result = apply_shell_input(
        create_initial_shell_state(),
        "/docs",
        handlers=ShellReadOnlyHandlers(docs=fail),
    )

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Unable to load document summary.",
    )


def test_apply_shell_input_trace_handler_error_appends_friendly_error() -> None:
    def fail() -> str:
        raise RuntimeError("broken trace")

    result = apply_shell_input(
        create_initial_shell_state(),
        "/trace",
        handlers=ShellReadOnlyHandlers(trace=fail),
    )

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Unable to load trace summary.",
    )


def test_apply_shell_input_settings_handler_error_appends_friendly_error() -> None:
    def fail() -> str:
        raise RuntimeError("broken settings")

    result = apply_shell_input(
        create_initial_shell_state(),
        "/settings",
        handlers=ShellReadOnlyHandlers(settings=fail),
    )

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Unable to load settings summary.",
    )


def test_apply_shell_input_handler_output_is_only_appended_as_text() -> None:
    state = apply_shell_input(create_initial_shell_state(), "question").state

    result = apply_shell_input(
        state,
        "/docs",
        handlers=ShellReadOnlyHandlers(docs=lambda: "/clear"),
    )

    assert len(result.state.messages) == len(state.messages) + 1
    assert result.state.messages[-1].text == "/clear"


def test_apply_shell_input_does_not_display_read_only_command_metadata() -> None:
    result = apply_shell_input(
        create_initial_shell_state(),
        "/docs",
        handlers=ShellReadOnlyHandlers(docs=lambda: "Workspace"),
    )

    rendered = format_transcript((result.state.messages[-1],))

    assert "Workspace" in rendered
    assert "shell_command" not in rendered
    assert "command" not in rendered


def test_read_only_handler_output_with_api_key_is_hidden_by_formatter() -> None:
    result = apply_shell_input(
        create_initial_shell_state(),
        "/settings",
        handlers=ShellReadOnlyHandlers(settings=lambda: "api_key: abc123"),
    )

    rendered = format_transcript(result.state.messages)

    assert "abc123" not in rendered
    assert "<hidden>" in rendered


def test_apply_shell_input_unknown_command_appends_error() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/unknown value")

    assert result.state.messages[-1] == TranscriptMessage(
        role="error",
        text="Unknown command: /unknown",
    )


def test_apply_shell_input_missing_arguments_appends_parser_error() -> None:
    result = apply_shell_input(create_initial_shell_state(), "/search")

    assert result.action == "none"
    assert result.search_query is None
    assert result.ask_question is None
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
    message = TranscriptMessage(role="tool", text="api_key=abc123")

    rendered = format_transcript((message,))

    assert "abc123" not in rendered
    assert "<hidden>" in rendered
