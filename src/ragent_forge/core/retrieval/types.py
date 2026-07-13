from typing import Literal

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
    if value == "bm25":
        return "bm25"
    if value == "semantic":
        return "semantic"
    if value == "hybrid":
        return "hybrid"
    return "lexical"
