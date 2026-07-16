from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeVar

ContextSelectionPolicy = Literal[
    "top_k_v1",
    "ranked_prefix_token_budget_v1",
    "ranked_query_fragment_budget_v1",
]

_Item = TypeVar("_Item")


def select_ranked_prefix_with_token_budget(
    items: Sequence[_Item],
    *,
    limit: int,
    max_context_tokens: int,
    characters_per_token: int,
    text_length: Callable[[_Item], int],
) -> list[_Item]:
    """Select a whole-item ranked prefix under a deterministic char budget."""

    if limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be greater than 0")
    if characters_per_token <= 0:
        raise ValueError("characters_per_token must be greater than 0")

    max_context_chars = max_context_tokens * characters_per_token
    selected: list[_Item] = []
    selected_chars = 0
    for item in items[:limit]:
        item_chars = text_length(item)
        if item_chars < 0:
            raise ValueError("item text length must be greater than or equal to 0")
        if selected_chars + item_chars > max_context_chars:
            break
        selected.append(item)
        selected_chars += item_chars
    return selected
