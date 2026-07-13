from typing import Literal, TypeGuard

RetrievalMode = Literal["lexical", "bm25", "semantic", "hybrid"]
RetrievalMethod = Literal[
    "lexical_token_overlap",
    "bm25",
    "semantic_cosine_similarity",
    "hybrid_rrf",
]

RETRIEVAL_MODES: tuple[RetrievalMode, ...] = (
    "lexical",
    "bm25",
    "semantic",
    "hybrid",
)


def normalize_retrieval_mode(value: str) -> RetrievalMode:
    normalized = value.strip().lower()
    if is_retrieval_mode(normalized):
        return normalized
    choices = ", ".join(RETRIEVAL_MODES)
    raise ValueError(
        f"Invalid retrieval mode: {value!r}. Expected one of: {choices}"
    )


def is_retrieval_mode(value: object) -> TypeGuard[RetrievalMode]:
    return isinstance(value, str) and value in RETRIEVAL_MODES
