from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from ragent_forge.tui.view_models import (
    AskPageState,
    SearchPageState,
    TuiAskStreamEvent,
    run_tui_ask,
    run_tui_search,
    stream_tui_ask,
)


def run_ask_worker(
    workspace_path: str | Path,
    question: str,
    mode: str,
    limit: int,
    max_context_chars: int,
    show_prompt: bool,
    *,
    stream: Callable[
        [str | Path, str, str, int, int, bool], Iterator[TuiAskStreamEvent]
    ] = stream_tui_ask,
    fallback: Callable[
        [str | Path, str, str, int, int, bool], AskPageState
    ] = run_tui_ask,
    on_delta: Callable[[str], None] | None = None,
) -> AskPageState:
    final_state: AskPageState | None = None
    for event in stream(
        workspace_path,
        question,
        mode,
        limit,
        max_context_chars,
        show_prompt,
    ):
        if event.type == "delta":
            if event.text and on_delta is not None:
                on_delta(event.text)
        elif event.type == "done":
            final_state = event.state
    if final_state is not None:
        return final_state
    return fallback(
        workspace_path,
        question,
        mode,
        limit,
        max_context_chars,
        show_prompt,
    )


def run_search_worker(
    workspace_path: str | Path,
    query: str,
    mode: str,
    limit: int,
    *,
    search: Callable[[str | Path, str, str, int], SearchPageState] = run_tui_search,
) -> SearchPageState:
    return search(workspace_path, query, mode, limit)
