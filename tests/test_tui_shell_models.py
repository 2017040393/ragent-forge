import pytest

from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.app.services.session_service import (
    TuiSession,
    TuiSessionMessage,
    TuiSessionRun,
    TuiSessionSource,
    TuiSessionTurn,
)
from ragent_forge.tui.shell_models import (
    WELCOME_MESSAGE,
    ShellState,
    TranscriptMessage,
    TranscriptRole,
    TranscriptSource,
    append_message,
    append_messages,
    clear_transcript,
    create_initial_shell_state,
    format_chat_transcript,
    format_conversation_transcript,
    format_shell_inspector,
    format_shell_source_details,
    format_shell_status,
    format_transcript,
    format_transcript_message,
    format_transcript_sources,
    message_from_search_results,
    message_from_search_state,
    messages_from_ask_state,
    replace_state_from_session,
    select_next_source,
    select_next_turn,
    select_previous_source,
    select_source,
    select_source_by_rank,
    select_turn_by_id,
    set_available_sources,
    set_limit,
    set_max_context_chars,
    set_retrieval_mode,
    set_running,
    set_show_prompt,
    transcript_sources_from_search_results,
)
from ragent_forge.tui.view_models import AskPageState, SearchPageState


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

    assert state.retrieval_mode == "hybrid"
    assert state.limit == 5
    assert state.max_context_chars == 4000
    assert state.show_prompt is False
    assert state.running is False
    assert state.notice is None
    assert state.selected_source is None
    assert state.available_sources == ()
    assert state.messages == ()
    assert state.current_session_id is None
    assert state.current_session_title is None
    assert state.selected_turn_id is None


def test_replace_state_from_session_loads_chat_turns_and_selects_latest() -> None:
    session = make_session()

    state = replace_state_from_session(create_initial_shell_state(), session)

    assert state.current_session_id == "session-1"
    assert state.current_session_title == "Agentic RAG chat"
    assert state.session_pinned is True
    assert state.session_starred is False
    assert state.selected_turn_id == "turn-2"
    assert state.available_sources[0].source_path == "/knowledge/source_2.md"
    assert state.messages == (
        TranscriptMessage(
            role="user",
            text="What is Agentic RAG?",
            turn_id="turn-1",
        ),
        TranscriptMessage(
            role="assistant",
            text="Agentic RAG plans before retrieval.",
            metadata={
                "operation": "ask",
                "retrieval_mode": "hybrid",
                "retrieval_method": "hybrid_rrf",
                "limit": 5,
                "max_context_chars": 4000,
                "show_prompt": False,
                "generation_status": "success",
                "generation_provider": "openai_responses",
                "source_count": 1,
            },
            sources=(
                TranscriptSource(
                    rank=1,
                    chunk_id="/knowledge/rag.md::chunk-0001",
                    source_path="/knowledge/source_1.md",
                    score=0.1,
                    preview="Preview 1",
                    metadata={"retrieval_method": "hybrid_rrf"},
                ),
            ),
            turn_id="turn-1",
        ),
        TranscriptMessage(role="user", text="And BM25?", turn_id="turn-2"),
        TranscriptMessage(
            role="assistant",
            text="BM25 is a lexical baseline.",
            metadata={
                "operation": "ask",
                "retrieval_mode": "bm25",
                "retrieval_method": "bm25",
                "limit": 3,
                "max_context_chars": 2000,
                "show_prompt": True,
                "generation_status": "not_configured",
                "generation_provider": "null",
                "source_count": 1,
            },
            sources=(
                TranscriptSource(
                    rank=2,
                    chunk_id="/knowledge/rag.md::chunk-0002",
                    source_path="/knowledge/source_2.md",
                    score=0.2,
                    preview="Preview 2",
                    metadata={"retrieval_method": "hybrid_rrf"},
                ),
            ),
            turn_id="turn-2",
        ),
    )


def test_select_turn_by_id_updates_sources_and_inspector() -> None:
    state = replace_state_from_session(create_initial_shell_state(), make_session())

    selected = select_turn_by_id(state, "turn-1")

    assert selected.selected_turn_id == "turn-1"
    assert selected.available_sources[0].source_path == "/knowledge/source_1.md"
    assert selected.selected_source == selected.available_sources[0]
    inspector = format_shell_inspector(selected)
    assert "Answer run" in inspector
    assert "turn: turn-1" in inspector
    assert "mode: hybrid" in inspector
    assert "source_1.md" in inspector


def test_select_next_turn_cycles_across_assistant_turns() -> None:
    state = replace_state_from_session(create_initial_shell_state(), make_session())

    wrapped = select_next_turn(state)

    assert wrapped.selected_turn_id == "turn-1"
    assert wrapped.selected_source is not None
    assert wrapped.selected_source.source_path == "/knowledge/source_1.md"


def test_append_message_returns_new_state_and_preserves_old_state() -> None:
    state = create_initial_shell_state()
    message = TranscriptMessage(role="user", text="What is Agentic RAG?")

    updated = append_message(state, message)

    assert updated is not state
    assert state.messages == ()
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
    assert updated.available_sources == (source,)


def test_append_message_with_sources_replaces_available_sources() -> None:
    selected = make_source(1)
    next_source = make_source(2)
    state = ShellState(
        selected_source=selected,
        available_sources=(selected,),
    )

    updated = append_message(
        state,
        TranscriptMessage(role="assistant", text="answer", sources=(next_source,)),
    )

    assert updated.selected_source == next_source
    assert updated.available_sources == (next_source,)


def test_append_message_without_sources_preserves_available_sources() -> None:
    source = make_source()
    state = ShellState(
        messages=(TranscriptMessage(role="system", text="hello"),),
        selected_source=source,
        available_sources=(source,),
    )

    updated = append_message(state, TranscriptMessage(role="tool", text="ok"))

    assert updated.selected_source == source
    assert updated.available_sources == (source,)


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
    assert updated.available_sources == (source,)


def test_clear_transcript_preserves_settings_and_resets_transcript() -> None:
    state = ShellState(
        retrieval_mode="semantic",
        limit=3,
        max_context_chars=1234,
        show_prompt=True,
        running=True,
        messages=(TranscriptMessage(role="assistant", text="old"),),
        selected_source=make_source(),
        available_sources=(make_source(),),
    )

    cleared = clear_transcript(state)

    assert cleared.retrieval_mode == "semantic"
    assert cleared.limit == 3
    assert cleared.max_context_chars == 1234
    assert cleared.show_prompt is True
    assert cleared.running is False
    assert cleared.selected_source is None
    assert cleared.available_sources == ()
    assert cleared.notice is None
    assert cleared.messages == ()


@pytest.mark.parametrize("mode", ["lexical", "bm25", "semantic", "hybrid"])
def test_set_retrieval_mode_accepts_supported_modes(mode: str) -> None:
    state = create_initial_shell_state()

    updated = set_retrieval_mode(state, mode)

    assert updated.retrieval_mode == mode
    assert updated.messages == state.messages


def test_set_retrieval_mode_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="Invalid retrieval mode: rerank"):
        set_retrieval_mode(create_initial_shell_state(), "rerank")


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


def test_set_available_sources_selects_first_source() -> None:
    first = make_source(1)
    second = make_source(2)

    updated = set_available_sources(create_initial_shell_state(), [first, second])

    assert updated.available_sources == (first, second)
    assert updated.selected_source == first


def test_set_available_sources_empty_clears_selected_source() -> None:
    source = make_source()
    state = ShellState(selected_source=source, available_sources=(source,))

    updated = set_available_sources(state, ())

    assert updated.available_sources == ()
    assert updated.selected_source is None


def test_select_source_by_rank_selects_first_and_second_source() -> None:
    first = make_source(1)
    second = make_source(2)
    state = ShellState(available_sources=(first, second), selected_source=first)

    assert select_source_by_rank(state, 1).selected_source == first
    assert select_source_by_rank(state, 2).selected_source == second


def test_select_source_by_rank_rejects_non_positive_rank() -> None:
    state = ShellState(available_sources=(make_source(1),))

    with pytest.raises(ValueError, match="Source rank must be a positive integer."):
        select_source_by_rank(state, 0)


def test_select_source_by_rank_rejects_out_of_range_rank() -> None:
    state = ShellState(available_sources=(make_source(1), make_source(2)))

    with pytest.raises(
        ValueError,
        match="Source rank out of range. Available sources: 1-2.",
    ):
        select_source_by_rank(state, 99)


def test_select_source_by_rank_requires_available_sources() -> None:
    with pytest.raises(
        ValueError,
        match="No sources available. Run /search <query> or ask a question first.",
    ):
        select_source_by_rank(create_initial_shell_state(), 1)


def test_select_next_source_moves_forward_and_wraps() -> None:
    first = make_source(1)
    second = make_source(2)
    state = ShellState(available_sources=(first, second), selected_source=first)

    moved = select_next_source(state)
    wrapped = select_next_source(moved)

    assert moved.selected_source == second
    assert wrapped.selected_source == first


def test_select_next_source_selects_first_when_none_selected() -> None:
    first = make_source(1)
    state = ShellState(available_sources=(first,), selected_source=None)

    assert select_next_source(state).selected_source == first


def test_select_previous_source_moves_backward_and_wraps() -> None:
    first = make_source(1)
    second = make_source(2)
    state = ShellState(available_sources=(first, second), selected_source=second)

    moved = select_previous_source(state)
    wrapped = select_previous_source(moved)

    assert moved.selected_source == first
    assert wrapped.selected_source == second


def test_select_previous_source_selects_first_when_none_selected() -> None:
    first = make_source(1)
    state = ShellState(available_sources=(first,), selected_source=None)

    assert select_previous_source(state).selected_source == first


def test_selecting_source_does_not_mutate_messages() -> None:
    source = make_source()
    message = TranscriptMessage(role="system", text="hello")
    state = ShellState(messages=(message,), available_sources=(source,))

    updated = select_source_by_rank(state, 1)

    assert updated.messages == state.messages


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


def make_session_source(rank: int = 1) -> TuiSessionSource:
    return TuiSessionSource(
        rank=rank,
        chunk_id=f"/knowledge/rag.md::chunk-{rank:04d}",
        source_path=f"/knowledge/source_{rank}.md",
        score=0.1 * rank,
        preview=f"Preview {rank}",
        metadata={"retrieval_method": "hybrid_rrf"},
    )


def make_session() -> TuiSession:
    return TuiSession(
        id="session-1",
        title="Agentic RAG chat",
        created_at="2026-07-08T00:00:00Z",
        updated_at="2026-07-08T00:00:01Z",
        pinned=True,
        starred=False,
        turns=(
            TuiSessionTurn(
                id="turn-1",
                created_at="2026-07-08T00:00:01Z",
                user_message=TuiSessionMessage(
                    role="user",
                    text="What is Agentic RAG?",
                    created_at="2026-07-08T00:00:01Z",
                ),
                assistant_message=TuiSessionMessage(
                    role="assistant",
                    text="Agentic RAG plans before retrieval.",
                    created_at="2026-07-08T00:00:02Z",
                ),
                sources=(make_session_source(1),),
                run=TuiSessionRun(
                    retrieval_mode="hybrid",
                    retrieval_method="hybrid_rrf",
                    limit=5,
                    max_context_chars=4000,
                    show_prompt=False,
                    generation_status="success",
                    generation_provider="openai_responses",
                ),
            ),
            TuiSessionTurn(
                id="turn-2",
                created_at="2026-07-08T00:00:03Z",
                user_message=TuiSessionMessage(
                    role="user",
                    text="And BM25?",
                    created_at="2026-07-08T00:00:03Z",
                ),
                assistant_message=TuiSessionMessage(
                    role="assistant",
                    text="BM25 is a lexical baseline.",
                    created_at="2026-07-08T00:00:04Z",
                ),
                sources=(make_session_source(2),),
                run=TuiSessionRun(
                    retrieval_mode="bm25",
                    retrieval_method="bm25",
                    limit=3,
                    max_context_chars=2000,
                    show_prompt=True,
                    generation_status="not_configured",
                    generation_provider="null",
                ),
            ),
        ),
    )


def test_format_shell_status_includes_notice_when_present() -> None:
    state = ShellState(notice="Vector index not found.")

    assert format_shell_status(state) == (
        "mode: hybrid | limit: 5 | context: 4000 | prompt: off | status: idle\n"
        "Vector index not found."
    )


def test_format_shell_inspector_without_selected_source_keeps_basic_details() -> None:
    state = ShellState(
        retrieval_mode="semantic",
        limit=5,
        max_context_chars=4000,
        show_prompt=False,
        messages=(TranscriptMessage(role="system", text="hello"),),
    )

    text = format_shell_inspector(state)

    assert text == "Inspector\n\nNo source selected."
    assert "Selected source" not in text


def test_format_shell_inspector_shows_notice_without_source() -> None:
    text = format_shell_inspector(
        ShellState(
            notice=(
                "Vector index not found.\n"
                "Run `ragent index build` first."
            )
        )
    )

    assert "Status" in text
    assert "Vector index not found." in text
    assert "ragent index build" in text
    assert "Shell details" not in text


def test_format_shell_inspector_with_selected_source_shows_source_details() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="/knowledge/rag.md::chunk-0001",
        source_path="/very/long/path/agentic_rag.md",
        score=0.832123,
        preview="Agentic RAG adds planning before retrieval.",
    )
    state = ShellState(
        retrieval_mode="semantic",
        messages=(TranscriptMessage(role="tool", text="answer", sources=(source,)),),
        selected_source=source,
    )

    text = format_shell_inspector(state)

    assert "Selected source" in text
    assert "Evidence" in text
    assert "Location" in text
    assert "Preview" in text
    assert "rank: 1" in text
    assert "source: agentic_rag.md" in text
    assert "chunk: chunk-0001" in text
    assert "score: 0.8321" in text
    assert "preview:" in text
    assert "  Agentic RAG adds planning before retrieval." in text
    assert "/very/long/path" not in text


def test_format_shell_source_details_highlights_query_terms() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="/knowledge/rag.md::chunk-0001",
        source_path="/knowledge/agentic_rag.md",
        score=0.832123,
        preview="Agentic RAG adds planning before retrieval.",
        metadata={
            "media_type": "application/pdf",
            "query": "agentic planning",
            "page_start": 3,
            "page_end": 3,
        },
    )

    text = format_shell_source_details(source)

    assert "source: agentic_rag.md p.3" in text
    assert "[[Agentic]] RAG adds [[planning]] before retrieval." in text


def test_format_shell_inspector_shows_allowlisted_retrieval_metadata() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="chunk-0001",
        source_path="/knowledge/agentic_rag.md",
        score=0.0325,
        preview="preview",
        metadata={
            "retrieval_method": "hybrid_rrf",
            "fusion_method": "reciprocal_rank_fusion",
            "matched_modes": ["bm25", "semantic"],
            "sparse_rank": 1,
            "dense_rank": 2,
            "hybrid_score": 0.0325,
        },
    )

    text = format_shell_inspector(ShellState(selected_source=source))

    assert "Retrieval metadata" in text
    assert "method: hybrid_rrf" in text
    assert "fusion: reciprocal_rank_fusion" in text
    assert "matched: bm25, semantic" in text
    assert "sparse_rank: 1" in text
    assert "dense_rank: 2" in text
    assert "hybrid_score: 0.0325" in text


def test_format_shell_inspector_filters_disallowed_source_metadata() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="chunk-0001",
        source_path="/knowledge/agentic_rag.md",
        score=0.5,
        preview="preview",
        metadata={
            "retrieval_method": "semantic",
            "api_key": "secret-value",
            "secret": "secret-value",
            "token": "token-value",
            "authorization": "Bearer abc123",
            "embedding": [1.0, 2.0, 3.0],
            "raw_internal_note": "do-not-show",
        },
    )

    text = format_shell_inspector(ShellState(selected_source=source))

    assert "method: semantic" in text
    assert "api_key" not in text
    assert "secret-value" not in text
    assert "token-value" not in text
    assert "Bearer abc123" not in text
    assert "[1.0, 2.0, 3.0]" not in text
    assert "raw_internal_note" not in text
    assert "do-not-show" not in text


def test_format_shell_source_details_caps_long_preview() -> None:
    long_preview = " ".join(f"word{i}" for i in range(80))
    source = TranscriptSource(
        rank=1,
        chunk_id="chunk-0001",
        source_path="/knowledge/agentic_rag.md",
        score=0.5,
        preview=long_preview,
    )

    text = format_shell_source_details(source)
    preview_text = text.split("preview:\n", 1)[1].split("\n\n", 1)[0]

    assert len(preview_text) <= 250
    assert "word79" not in preview_text
    assert "..." in preview_text


def test_format_shell_source_details_preserves_preview_indentation() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="chunk-0001",
        source_path="/knowledge/agentic_rag.md",
        score=0.5,
        preview="first line\n  nested line\n    deeper line",
    )

    text = format_shell_source_details(source)

    assert "  first line" in text
    assert "    nested line" in text
    assert "      deeper line" in text


def test_format_shell_source_details_shows_allowlisted_retrieval_metadata() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="chunk-0001",
        source_path="/knowledge/agentic_rag.md",
        score=0.5,
        preview="preview",
        metadata={
            "retrieval_method": "hybrid_rrf",
            "fusion_method": "reciprocal_rank_fusion",
            "matched_modes": ["bm25", "semantic"],
            "sparse_rank": 1,
            "dense_rank": 2,
            "hybrid_score": 0.0325,
        },
    )

    text = format_shell_source_details(source)

    assert "method: hybrid_rrf" in text
    assert "fusion: reciprocal_rank_fusion" in text
    assert "matched: bm25, semantic" in text
    assert "sparse_rank: 1" in text
    assert "dense_rank: 2" in text
    assert "hybrid_score: 0.0325" in text


def test_format_shell_source_details_does_not_show_disallowed_metadata() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="chunk-0001",
        source_path="/knowledge/agentic_rag.md",
        score=0.5,
        preview="preview",
        metadata={
            "retrieval_method": "semantic",
            "api_key": "secret-value",
            "token": "token-value",
            "embedding": [1.0, 2.0, 3.0],
            "raw_internal_note": "do-not-show",
        },
    )

    text = format_shell_source_details(source)

    assert "method: semantic" in text
    assert "secret-value" not in text
    assert "token-value" not in text
    assert "[1.0, 2.0, 3.0]" not in text
    assert "raw_internal_note" not in text
    assert "do-not-show" not in text


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
    role: TranscriptRole,
    heading: str,
) -> None:
    text = format_transcript_message(TranscriptMessage(role=role, text="hello"))

    assert text == f"{heading}\n  hello"


def test_format_transcript_message_indents_multiline_text() -> None:
    message = TranscriptMessage(role="assistant", text="line 1\nline 2")

    text = format_transcript_message(message)

    assert text == "Assistant:\n  line 1\n  line 2"


def test_format_conversation_transcript_only_renders_user_and_assistant() -> None:
    source = make_source()
    text = format_conversation_transcript(
        (
            TranscriptMessage(role="system", text=WELCOME_MESSAGE),
            TranscriptMessage(role="tool", text="Running ask for: question"),
            TranscriptMessage(role="error", text="Vector index not found."),
            TranscriptMessage(role="user", text="question"),
            TranscriptMessage(role="assistant", text="answer", sources=(source,)),
        )
    )

    assert text == "User:\n  question\n\nAssistant: [1 source]\n  answer"
    assert "System:" not in text
    assert "Tool:" not in text
    assert "Error:" not in text
    assert "Sources:" not in text


def test_format_chat_transcript_hides_operational_command_output() -> None:
    source = make_source()

    text = format_chat_transcript(
        (
            TranscriptMessage(role="system", text=WELCOME_MESSAGE),
            TranscriptMessage(role="tool", text="Search results loaded."),
            TranscriptMessage(role="error", text="Trace load failed."),
            TranscriptMessage(role="user", text="What is Agentic RAG?"),
            TranscriptMessage(
                role="assistant",
                text="Agentic RAG plans retrieval steps.",
                metadata={"retrieval_mode": "hybrid", "source_count": 1},
                sources=(source,),
            ),
        )
    )

    assert text == (
        "User:\n"
        "  What is Agentic RAG?\n\n"
        "Assistant: [1 source]\n"
        "  Agentic RAG plans retrieval steps."
    )
    assert "Search results loaded" not in text
    assert "Trace load failed" not in text
    assert "retrieval_mode" not in text
    assert "Sources:" not in text


def test_format_chat_transcript_marks_failed_assistant_answer() -> None:
    text = format_chat_transcript(
        (
            TranscriptMessage(role="user", text="Question?"),
            TranscriptMessage(
                role="assistant",
                text="Ask failed.",
                metadata={"generation_status": "failed"},
            ),
        )
    )

    assert text == "User:\n  Question?\n\nAssistant: [failed]\n  Ask failed."


def test_format_transcript_message_normalizes_assistant_latex_math() -> None:
    message = TranscriptMessage(
        role="assistant",
        text=(
            r"High-dimensional probability studies \\(R^n\\), "
            r"\(\mathbb{R}^n\), \(\alpha \leq \beta\), and \(x_i^2\)."
        ),
    )

    text = format_transcript_message(message)

    assert text == (
        "Assistant:\n"
        "  High-dimensional probability studies R\u207f, "
        "\u211d\u207f, \u03b1 \u2264 \u03b2, and x\u1d62\u00b2."
    )


def test_format_transcript_message_normalizes_plain_superscript_math() -> None:
    message = TranscriptMessage(
        role="assistant",
        text=r"Use R^n, x^2, and \mathbb{R}^d as readable formulas.",
    )

    text = format_transcript_message(message)

    assert text == (
        "Assistant:\n"
        "  Use R\u207f, x\u00b2, and \u211d\u1d48 as readable formulas."
    )


def test_format_transcript_message_with_sources_appends_source_block() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="/knowledge/rag.md::chunk-0001",
        source_path="/very/long/path/agentic_rag.md",
        score=0.832123,
        preview="Agentic RAG adds planning before retrieval.",
        metadata={"retrieval_method": "semantic"},
    )
    message = TranscriptMessage(
        role="tool",
        text="Search results for: GAG\nResults: 1 | mode: semantic",
        sources=(source,),
    )

    text = format_transcript_message(message)

    assert "Tool:\n  Search results for: GAG\n  Results: 1 | mode: semantic" in text
    assert "\n\nSources:\n" in text
    assert "1. agentic_rag.md" in text
    assert "chunk=chunk-0001" in text
    assert "score=0.8321" in text
    assert "/very/long/path" not in text
    assert "retrieval_method" not in text


def test_format_transcript_message_source_block_hides_sensitive_metadata() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="chunk-0001",
        source_path="/knowledge/rag.md",
        score=1.0,
        preview="preview",
        metadata={
            "api_key": "secret-value",
            "token": "token-value",
            "embedding": [1.0, 2.0, 3.0],
        },
    )
    message = TranscriptMessage(role="tool", text="answer", sources=(source,))

    text = format_transcript_message(message)

    assert "Sources:" in text
    assert "secret-value" not in text
    assert "token-value" not in text
    assert "[1.0, 2.0, 3.0]" not in text


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


def test_format_transcript_includes_source_blocks_only_when_sources_exist() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="/knowledge/rag.md::chunk-0001",
        source_path="/knowledge/agentic_rag.md",
        score=0.8321,
        preview="Agentic RAG adds planning before retrieval.",
    )

    text = format_transcript(
        (
            TranscriptMessage(role="user", text="question"),
            TranscriptMessage(role="tool", text="answer", sources=(source,)),
        )
    )

    assert text.count("Sources:") == 1
    assert "User:\n  question\n\nTool:\n  answer\n\nSources:" in text


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


def test_format_transcript_sources_aligns_compact_source_labels() -> None:
    sources = (
        TranscriptSource(
            rank=1,
            chunk_id="/knowledge/agentic_rag.md::chunk-0000",
            source_path="/very/long/path/agentic_rag.md",
            score=2.0,
            preview="preview",
        ),
        TranscriptSource(
            rank=2,
            chunk_id="/knowledge/rag_basics.md::chunk-0000",
            source_path="/very/long/path/rag_basics.md",
            score=1.0,
            preview="preview",
        ),
    )

    lines = format_transcript_sources(sources).splitlines()

    assert lines == [
        "Sources:",
        "1. agentic_rag.md  score=2  chunk=chunk-0000",
        "2. rag_basics.md   score=1  chunk=chunk-0000",
        "",
        "Use /source <rank>, /source next, or /source prev to inspect evidence.",
    ]


def test_format_transcript_sources_includes_source_navigation_hint() -> None:
    text = format_transcript_sources((make_source(1),))

    assert (
        "Use /source <rank>, /source next, or /source prev to inspect evidence."
        in text
    )


def test_format_transcript_sources_truncates_very_long_compact_labels() -> None:
    long_name = "this_is_a_really_long_source_label_that_should_not_break_layout.md"
    source = TranscriptSource(
        rank=1,
        chunk_id="/knowledge/rag.md::chunk-0000",
        source_path=f"/very/long/path/{long_name}",
        score=0.0325,
        preview="preview",
    )

    line = format_transcript_sources((source,)).splitlines()[1]
    label = line.split(". ", 1)[1].split("  score=", 1)[0].rstrip()

    assert len(label) <= 40
    assert label.endswith("...")
    assert long_name not in line
    assert "/very/long/path" not in line


def test_format_transcript_sources_does_not_show_metadata_or_secrets() -> None:
    source = TranscriptSource(
        rank=1,
        chunk_id="/knowledge/rag.md::chunk-0000",
        source_path="/knowledge/rag.md",
        score=0.0325,
        preview="preview",
        metadata={
            "retrieval_method": "lexical",
            "api_key": "secret-value",
            "secret": "secret-value",
            "token": "token-value",
            "embedding": [1.0, 2.0, 3.0],
            "raw_internal_note": "do-not-show",
        },
    )

    text = format_transcript_sources((source,))

    assert "rag.md" in text
    assert "secret-value" not in text
    assert "token-value" not in text
    assert "[1.0, 2.0, 3.0]" not in text
    assert "raw_internal_note" not in text
    assert "do-not-show" not in text


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
    assert messages[0].sources == ()


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
        "retrieved_context_chars": 43,
        "estimated_context_tokens": 11,
        "prompt_preview_available": False,
    }
    assert len(message.sources) == 1


def test_ask_answer_metadata_feeds_selected_turn_inspector() -> None:
    result = make_search_result(text="Agentic RAG adds planning.")
    state = AskPageState(
        question="What is RAG?",
        retrieval_mode="hybrid",
        answer="Agentic RAG adds planning.",
        sources=[result],
        generation_status="success",
        generation_provider="openai_responses",
        prompt_preview="Use only retrieved context.",
        show_prompt=True,
        has_run=True,
    )
    message = messages_from_ask_state(state)[0]
    message = TranscriptMessage(
        role=message.role,
        text=message.text,
        metadata=message.metadata,
        sources=message.sources,
        turn_id="turn-1",
    )
    shell_state = ShellState(
        messages=(message,),
        selected_turn_id="turn-1",
    )

    inspector = format_shell_inspector(shell_state)
    transcript = format_chat_transcript((message,))

    assert "context chars: 26" in inspector
    assert "estimated tokens: 7" in inspector
    assert "prompt preview: available" in inspector
    assert "sources: 1" in inspector
    assert "context chars" not in transcript
    assert "estimated tokens" not in transcript


def test_messages_from_ask_state_with_answer_renders_sources_block() -> None:
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

    rendered = format_transcript(messages_from_ask_state(state))

    assert "Assistant:\n  Agentic RAG adds planning." in rendered
    assert "Sources:" in rendered
    assert "1. rag_basics.md" in rendered
    assert "chunk=chunk-0000" in rendered


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


def test_messages_from_ask_state_with_status_renders_sources_block() -> None:
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

    rendered = format_transcript(messages_from_ask_state(state))

    assert "Tool:" in rendered
    assert "Generation: not configured. Showing retrieved context only." in rendered
    assert "Sources:" in rendered
    assert "1. rag_basics.md" in rendered


def test_messages_from_ask_state_does_not_dump_prompt_preview() -> None:
    state = AskPageState(
        question="What is RAG?",
        retrieval_mode="lexical",
        answer="Short answer.",
        prompt_preview="PROMPT PREVIEW SHOULD NOT APPEAR",
        show_prompt=True,
        has_run=True,
    )

    rendered = format_transcript(messages_from_ask_state(state))

    assert "Short answer." in rendered
    assert "PROMPT PREVIEW SHOULD NOT APPEAR" not in rendered


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


def test_message_from_search_state_with_results_returns_tool_message() -> None:
    result = make_search_result()
    state = SearchPageState(
        query="agentic rag",
        retrieval_mode="hybrid",
        limit=3,
        results=[result],
        has_searched=True,
    )

    message = message_from_search_state(state)

    assert message.role == "tool"
    assert message.text == "Search results for: agentic rag\nResults: 1 | mode: hybrid"
    assert message.metadata == {
        "operation": "search",
        "query": "agentic rag",
        "retrieval_mode": "hybrid",
        "limit": 3,
        "result_count": 1,
    }
    assert len(message.sources) == 1
    assert message.sources[0].source_path == "/very/long/path/rag_basics.md"


def test_message_from_search_state_with_no_results_returns_no_matches_message() -> None:
    state = SearchPageState(
        query="missing",
        retrieval_mode="lexical",
        limit=5,
        has_searched=True,
    )

    message = message_from_search_state(state)

    assert message.role == "tool"
    assert message.text == "No matches found. Try another query or retrieval mode."
    assert message.metadata == {
        "operation": "search",
        "query": "missing",
        "retrieval_mode": "lexical",
        "limit": 5,
        "result_count": 0,
    }
    assert message.sources == ()


def test_message_from_search_state_with_error_returns_error_message() -> None:
    state = SearchPageState(
        query="agentic rag",
        retrieval_mode="semantic",
        limit=4,
        error="Vector index not found. Run `ragent index build` first.",
        has_searched=True,
    )

    message = message_from_search_state(state)

    assert message == TranscriptMessage(
        role="error",
        text="Vector index not found. Run `ragent index build` first.",
        metadata={
            "operation": "search",
            "query": "agentic rag",
            "retrieval_mode": "semantic",
            "limit": 4,
        },
    )


def test_message_from_search_state_formatter_hides_sensitive_source_metadata() -> None:
    result = make_search_result(
        metadata={
            "retrieval_method": "lexical",
            "api_key": "secret-value",
            "embedding": [1.0, 2.0, 3.0],
        },
    )
    state = SearchPageState(
        query="agentic rag",
        retrieval_mode="lexical",
        limit=1,
        results=[result],
        has_searched=True,
    )

    message = message_from_search_state(state)
    rendered = "\n".join(
        [
            format_transcript((message,)),
            format_transcript_sources(message.sources),
        ]
    )

    assert "secret-value" not in rendered
    assert "api_key" not in message.sources[0].metadata
    assert "embedding" not in message.sources[0].metadata
    assert "[1.0, 2.0, 3.0]" not in rendered


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
