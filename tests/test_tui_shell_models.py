import pytest

from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.tui.shell_models import (
    WELCOME_MESSAGE,
    ShellState,
    TranscriptMessage,
    TranscriptSource,
    append_message,
    append_messages,
    clear_transcript,
    create_initial_shell_state,
    format_shell_status,
    format_transcript,
    format_transcript_message,
    format_transcript_sources,
    message_from_search_results,
    messages_from_ask_state,
    select_source,
    set_limit,
    set_max_context_chars,
    set_retrieval_mode,
    set_running,
    set_show_prompt,
    transcript_sources_from_search_results,
)
from ragent_forge.tui.view_models import AskPageState


def make_search_result(
    *,
    chunk_id: str = "/knowledge/rag.md::chunk-0000",
    source_path: str = "/very/long/path/rag_basics.md",
    score: float = 0.0325,
    text: str = "Agentic RAG adds planning before retrieval.",
    metadata: dict[str, object] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc",
        source_path=source_path,
        start_char=0,
        end_char=42,
        score=score,
        text=text,
        metadata=metadata or {"retrieval_method": "lexical_token_overlap"},
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


def test_initial_shell_state_has_default_settings_and_welcome_message() -> None:
    state = create_initial_shell_state()

    assert state.retrieval_mode == "lexical"
    assert state.limit == 5
    assert state.max_context_chars == 4000
    assert state.show_prompt is False
    assert state.running is False
    assert state.selected_source is None
    assert state.messages == (
        TranscriptMessage(role="system", text=WELCOME_MESSAGE),
    )


def test_append_message_returns_new_state_and_preserves_old_state() -> None:
    state = create_initial_shell_state()
    message = TranscriptMessage(role="user", text="What is Agentic RAG?")

    updated = append_message(state, message)

    assert updated is not state
    assert state.messages == (TranscriptMessage(role="system", text=WELCOME_MESSAGE),)
    assert updated.messages[-1] == message


def test_append_message_preserves_settings() -> None:
    state = ShellState(
        retrieval_mode="hybrid",
        limit=8,
        max_context_chars=2000,
        show_prompt=True,
        running=True,
        messages=(TranscriptMessage(role="system", text="hello"),),
    )

    updated = append_message(state, TranscriptMessage(role="tool", text="ok"))

    assert updated.retrieval_mode == "hybrid"
    assert updated.limit == 8
    assert updated.max_context_chars == 2000
    assert updated.show_prompt is True
    assert updated.running is True


def test_append_message_selects_first_source_when_none_selected() -> None:
    source = make_source()
    state = create_initial_shell_state()

    updated = append_message(
        state,
        TranscriptMessage(role="assistant", text="answer", sources=(source,)),
    )

    assert updated.selected_source == source


def test_append_message_preserves_existing_selected_source() -> None:
    selected = make_source(1)
    next_source = make_source(2)
    state = ShellState(selected_source=selected)

    updated = append_message(
        state,
        TranscriptMessage(role="assistant", text="answer", sources=(next_source,)),
    )

    assert updated.selected_source == selected


def test_append_messages_preserves_order_and_selects_first_available_source() -> None:
    source = make_source()
    state = create_initial_shell_state()
    messages = (
        TranscriptMessage(role="user", text="question"),
        TranscriptMessage(role="assistant", text="answer", sources=(source,)),
    )

    updated = append_messages(state, messages)

    assert updated.messages[-2:] == messages
    assert updated.selected_source == source


def test_clear_transcript_preserves_settings_and_resets_transcript() -> None:
    state = ShellState(
        retrieval_mode="semantic",
        limit=3,
        max_context_chars=1234,
        show_prompt=True,
        running=True,
        messages=(TranscriptMessage(role="assistant", text="old"),),
        selected_source=make_source(),
    )

    cleared = clear_transcript(state)

    assert cleared.retrieval_mode == "semantic"
    assert cleared.limit == 3
    assert cleared.max_context_chars == 1234
    assert cleared.show_prompt is True
    assert cleared.running is False
    assert cleared.selected_source is None
    assert cleared.messages == (TranscriptMessage(role="system", text=WELCOME_MESSAGE),)


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_set_retrieval_mode_accepts_supported_modes(mode: str) -> None:
    state = create_initial_shell_state()

    updated = set_retrieval_mode(state, mode)

    assert updated.retrieval_mode == mode
    assert updated.messages == state.messages


def test_set_retrieval_mode_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="Invalid retrieval mode: bm25"):
        set_retrieval_mode(create_initial_shell_state(), "bm25")


def test_set_limit_accepts_positive_int() -> None:
    state = create_initial_shell_state()

    updated = set_limit(state, 10)

    assert updated.limit == 10
    assert updated.messages == state.messages


@pytest.mark.parametrize("limit", [0, -1])
def test_set_limit_rejects_non_positive_values(limit: int) -> None:
    with pytest.raises(ValueError, match="Limit must be positive."):
        set_limit(create_initial_shell_state(), limit)


def test_set_max_context_chars_accepts_positive_int() -> None:
    state = create_initial_shell_state()

    updated = set_max_context_chars(state, 1200)

    assert updated.max_context_chars == 1200
    assert updated.messages == state.messages


@pytest.mark.parametrize("value", [0, -1])
def test_set_max_context_chars_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError, match="Max context chars must be positive."):
        set_max_context_chars(create_initial_shell_state(), value)


def test_set_show_prompt_toggles_bool() -> None:
    state = create_initial_shell_state()

    enabled = set_show_prompt(state, True)
    disabled = set_show_prompt(enabled, False)

    assert enabled.show_prompt is True
    assert disabled.show_prompt is False
    assert disabled.messages == state.messages


def test_set_running_toggles_running() -> None:
    state = create_initial_shell_state()

    running = set_running(state, True)
    idle = set_running(running, False)

    assert running.running is True
    assert idle.running is False
    assert idle.messages == state.messages


def test_select_source_sets_and_clears_source() -> None:
    source = make_source()
    state = create_initial_shell_state()

    selected = select_source(state, source)
    cleared = select_source(selected, None)

    assert selected.selected_source == source
    assert cleared.selected_source is None
    assert cleared.messages == state.messages


def test_format_shell_status_reflects_settings_and_running_state() -> None:
    state = ShellState(
        retrieval_mode="hybrid",
        limit=7,
        max_context_chars=999,
        show_prompt=True,
        running=True,
    )

    assert format_shell_status(state) == (
        "mode: hybrid | limit: 7 | context: 999 | prompt: on | status: running"
    )


@pytest.mark.parametrize(
    ("role", "heading"),
    [
        ("system", "System:"),
        ("user", "User:"),
        ("assistant", "Assistant:"),
        ("tool", "Tool:"),
        ("error", "Error:"),
    ],
)
def test_format_transcript_message_formats_all_roles(
    role: str,
    heading: str,
) -> None:
    text = format_transcript_message(TranscriptMessage(role=role, text="hello"))

    assert text == f"{heading}\n  hello"


def test_format_transcript_message_indents_multiline_text() -> None:
    message = TranscriptMessage(role="assistant", text="line 1\nline 2")

    text = format_transcript_message(message)

    assert text == "Assistant:\n  line 1\n  line 2"


def test_format_transcript_message_preserves_normal_token_terms() -> None:
    message = TranscriptMessage(
        role="assistant",
        text=(
            "Tokenization splits context tokens.\n"
            "Embedding token counts are estimates."
        ),
    )

    text = format_transcript_message(message)

    assert "Tokenization splits context tokens." in text
    assert "Embedding token counts are estimates." in text
    assert "<hidden>" not in text


@pytest.mark.parametrize(
    "line",
    [
        "api_key=abc123",
        "api_key: abc123",
        "authorization: Bearer abc123",
        "bearer abc123",
        "secret=abc123",
        "secret: abc123",
        "token=abc123",
        "token: abc123",
    ],
)
def test_format_transcript_message_hides_credential_shaped_lines(line: str) -> None:
    text = format_transcript_message(TranscriptMessage(role="assistant", text=line))

    assert text == "Assistant:\n  <hidden>"
    assert "abc123" not in text


def test_format_transcript_joins_messages_with_blank_lines() -> None:
    messages = (
        TranscriptMessage(role="user", text="question"),
        TranscriptMessage(role="assistant", text="answer"),
    )

    text = format_transcript(messages)

    assert text == "User:\n  question\n\nAssistant:\n  answer"


def test_format_transcript_handles_empty_transcript() -> None:
    assert format_transcript(()) == ""


def test_format_transcript_sources_uses_compact_labels() -> None:
    sources = (
        TranscriptSource(
            rank=1,
            chunk_id="/knowledge/rag.md::chunk-0000",
            source_path="/very/long/path/rag_basics.md",
            score=0.0325,
            preview="Agentic RAG adds planning.",
        ),
    )

    text = format_transcript_sources(sources)

    assert "Sources:" in text
    assert "1. rag_basics.md" in text
    assert "score=0.0325" in text
    assert "chunk=chunk-0000" in text
    assert "/very/long/path" not in text


def test_format_transcript_sources_handles_empty_sources() -> None:
    assert format_transcript_sources(()) == "Sources:\nNo sources."


def test_transcript_sources_from_search_results_preserves_order_and_metadata() -> None:
    first = make_search_result(
        chunk_id="doc::chunk-0000",
        source_path="/knowledge/rag.md",
        score=1.0,
        metadata={
            "retrieval_method": "lexical",
            "api_key": "not-carried",
            "embedding": [1.0, 2.0, 3.0],
        },
    )
    second = make_search_result(
        chunk_id="doc::chunk-0001",
        source_path="/knowledge/agentic.md",
        score=0.5,
        metadata={"retrieval_method": "semantic"},
    )

    sources = transcript_sources_from_search_results([first, second])
    first.metadata["retrieval_method"] = "mutated"

    assert [source.rank for source in sources] == [1, 2]
    assert [source.chunk_id for source in sources] == [
        "doc::chunk-0000",
        "doc::chunk-0001",
    ]
    assert sources[0].source_path == "/knowledge/rag.md"
    assert sources[0].score == 1.0
    assert sources[0].metadata["retrieval_method"] == "lexical"
    assert "api_key" not in sources[0].metadata
    assert "embedding" not in sources[0].metadata
    assert "Agentic RAG" in sources[0].preview


def test_messages_from_ask_state_with_error_returns_error_message() -> None:
    state = AskPageState(
        question="What is RAG?",
        retrieval_mode="semantic",
        error="Vector index not found.",
        has_run=True,
    )

    messages = messages_from_ask_state(state)

    assert messages == (
        TranscriptMessage(
            role="error",
            text="Vector index not found.",
            metadata={"operation": "ask", "retrieval_mode": "semantic"},
        ),
    )


def test_messages_from_ask_state_with_answer_returns_assistant_message() -> None:
    result = make_search_result()
    state = AskPageState(
        question="What is RAG?",
        retrieval_mode="hybrid",
        answer="Agentic RAG adds planning.",
        sources=[result],
        generation_status="success",
        generation_provider="openai_responses",
        has_run=True,
    )

    messages = messages_from_ask_state(state)

    assert len(messages) == 1
    message = messages[0]
    assert message.role == "assistant"
    assert message.text == "Agentic RAG adds planning."
    assert message.metadata == {
        "operation": "ask",
        "retrieval_mode": "hybrid",
        "generation_status": "success",
        "generation_provider": "openai_responses",
        "source_count": 1,
    }
    assert len(message.sources) == 1


def test_messages_from_ask_state_with_status_returns_tool_message() -> None:
    result = make_search_result()
    state = AskPageState(
        question="What is RAG?",
        retrieval_mode="lexical",
        status="Generation: not configured. Showing retrieved context only.",
        sources=[result],
        generation_status="not_configured",
        generation_provider="null",
        has_run=True,
    )

    messages = messages_from_ask_state(state)

    assert messages[0].role == "tool"
    assert messages[0].text == (
        "Generation: not configured. Showing retrieved context only."
    )
    assert messages[0].metadata["source_count"] == 1
    assert len(messages[0].sources) == 1


def test_messages_from_ask_state_fallback_returns_tool_message() -> None:
    state = AskPageState(
        question="What is RAG?",
        retrieval_mode="lexical",
        status=None,
        has_run=True,
    )

    messages = messages_from_ask_state(state)

    assert messages == (
        TranscriptMessage(
            role="tool",
            text="Ask completed.",
            metadata={"operation": "ask", "retrieval_mode": "lexical"},
        ),
    )


def test_message_from_search_results_returns_tool_message_with_metadata() -> None:
    result = make_search_result()

    message = message_from_search_results(
        "agentic rag",
        "hybrid",
        [result],
    )

    assert message.role == "tool"
    assert message.text == "Search results for: agentic rag\nResults: 1 | mode: hybrid"
    assert message.metadata == {
        "operation": "search",
        "query": "agentic rag",
        "retrieval_mode": "hybrid",
        "result_count": 1,
    }
    assert len(message.sources) == 1


def test_message_from_search_results_handles_no_results() -> None:
    message = message_from_search_results("missing", "lexical", [])

    assert message.role == "tool"
    assert message.text == "No matches found. Try another query or retrieval mode."
    assert message.metadata["result_count"] == 0
    assert message.sources == ()


def test_formatting_helpers_do_not_display_metadata_or_api_keys() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="chunk-0000",
        source_path="/knowledge/rag.md",
        score=1.0,
        preview="preview",
        metadata={
            "api_key": "secret-value",
            "token": "token-value",
            "embedding": [1.0, 2.0, 3.0],
        },
    )
    message = TranscriptMessage(
        role="assistant",
        text="answer",
        metadata={"api_key": "secret-value"},
        sources=(source,),
    )

    rendered = "\n".join(
        [
            format_transcript_message(message),
            format_transcript((message,)),
            format_transcript_sources((source,)),
        ]
    )

    assert "secret-value" not in rendered
    assert "api_key" not in rendered
    assert "token-value" not in rendered
    assert "[1.0, 2.0, 3.0]" not in rendered


def test_helpers_do_not_mutate_input_state() -> None:
    source = make_source()
    original = ShellState(
        retrieval_mode="lexical",
        limit=5,
        max_context_chars=4000,
        show_prompt=False,
        running=False,
        messages=(TranscriptMessage(role="system", text="hello"),),
        selected_source=None,
    )

    _ = append_message(
        original,
        TranscriptMessage(role="assistant", text="answer", sources=(source,)),
    )
    _ = set_limit(original, 10)
    _ = clear_transcript(original)

    assert original == ShellState(
        retrieval_mode="lexical",
        limit=5,
        max_context_chars=4000,
        show_prompt=False,
        running=False,
        messages=(TranscriptMessage(role="system", text="hello"),),
        selected_source=None,
    )
