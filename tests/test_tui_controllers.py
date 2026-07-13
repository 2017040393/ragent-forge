from collections.abc import Iterator
from pathlib import Path

from ragent_forge.tui.controllers.session import (
    assistant_message_from_ask_result,
    session_run_from_ask_result,
)
from ragent_forge.tui.controllers.workers import (
    run_ask_worker,
    run_search_worker,
)
from ragent_forge.tui.view_models import (
    AskPageState,
    SearchPageState,
    TuiAskStreamEvent,
)


def test_ask_worker_coordinates_stream_deltas_and_final_state() -> None:
    deltas: list[str] = []
    expected = AskPageState(
        question="What is RAG?",
        answer="Retrieval augmented generation.",
        has_run=True,
        trace_id="ask-retrieval-controller-trace",
    )

    def stream(
        _workspace: str | Path,
        _question: str,
        _mode: str,
        _limit: int,
        _max_context_chars: int,
        _show_prompt: bool,
    ) -> Iterator[TuiAskStreamEvent]:
        yield TuiAskStreamEvent(type="delta", text="Retrieval ")
        yield TuiAskStreamEvent(type="delta", text="augmented generation.")
        yield TuiAskStreamEvent(type="done", state=expected)

    def forbidden_fallback(
        _workspace: str | Path,
        _question: str,
        _mode: str,
        _limit: int,
        _max_context_chars: int,
        _show_prompt: bool,
    ) -> AskPageState:
        raise AssertionError("completed streams must not call the fallback")

    result = run_ask_worker(
        ".ragent",
        "What is RAG?",
        "lexical",
        5,
        4000,
        False,
        stream=stream,
        fallback=forbidden_fallback,
        on_delta=deltas.append,
    )

    assert result is expected
    assert deltas == ["Retrieval ", "augmented generation."]


def test_search_worker_uses_injected_search_boundary() -> None:
    expected = SearchPageState(query="memory", has_searched=True)

    def search(
        workspace: str | Path,
        query: str,
        mode: str,
        limit: int,
    ) -> SearchPageState:
        assert Path(workspace) == Path(".ragent")
        assert (query, mode, limit) == ("memory", "bm25", 3)
        return expected

    assert run_search_worker(
        ".ragent",
        "memory",
        "bm25",
        3,
        search=search,
    ) is expected


def test_session_controller_preserves_trace_reference() -> None:
    state = AskPageState(
        question="What is saved?",
        retrieval_mode="lexical",
        answer="A persisted answer.",
        generation_status="not_configured",
        generation_provider="null",
        has_run=True,
        trace_id="ask-retrieval-session-trace",
    )

    message = assistant_message_from_ask_result(state)
    run = session_run_from_ask_result(state, message)

    assert message.metadata["trace_id"] == state.trace_id
    assert run.trace_id == state.trace_id
    assert run.retrieval_method == "lexical_token_overlap"
