from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragent_forge.app.models import AppConfig
from ragent_forge.app.ports import ApplicationWorkspace
from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.embedding_service import EmbeddingService
from ragent_forge.app.services.generation_service import GenerationService
from ragent_forge.app.services.hybrid_search_service import (
    HybridDenseMethod,
    HybridSearchConfig,
    HybridSearchService,
    HybridSparseMethod,
)
from ragent_forge.app.services.retrieval_pipeline_service import (
    RetrievalEngine,
)
from ragent_forge.app.services.search_service import (
    BM25SearchService,
    LexicalSearchService,
    SearchResult,
)
from ragent_forge.app.services.semantic_search_service import SemanticSearchService
from ragent_forge.app.services.text_generation_client import (
    OpenAIResponsesTextGenerationClient,
)
from ragent_forge.core.retrieval.types import RetrievalMethod, RetrievalMode
from ragent_forge.infrastructure.http_client import default_http_client


class SearchService(Protocol):
    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        ...

    def count_chunks(self) -> int:
        ...


@dataclass(frozen=True)
class RetrievalRuntime:
    search_service: SearchService
    retrieval_engine: RetrievalEngine
    retrieval_mode: RetrievalMode
    retrieval_method: RetrievalMethod
    embedding_provider: str | None = None
    embedding_model: str | None = None
    index_path: Path | None = None
    fusion_method: str | None = None
    rrf_k: int | None = None
    sparse_method: HybridSparseMethod | None = None
    dense_method: HybridDenseMethod | None = None
    sparse_weight: float | None = None
    dense_weight: float | None = None
    lexical_weight: float | None = None
    semantic_weight: float | None = None
    candidate_limit: int | None = None

    @property
    def retrieval_pipeline(self) -> RetrievalEngine:
        """Compatibility alias for the pre-engine composition contract."""
        return self.retrieval_engine


def build_retrieval_runtime(
    workspace: ApplicationWorkspace,
    mode: RetrievalMode,
    *,
    limit: int,
    config: AppConfig | None = None,
) -> RetrievalRuntime:
    if mode == "lexical":
        search_service = LexicalSearchService(workspace)
        return RetrievalRuntime(
            search_service=search_service,
            retrieval_engine=RetrievalEngine(
                search_service,
                mode,
                "lexical_token_overlap",
                _snapshot_id(workspace),
            ),
            retrieval_mode=mode,
            retrieval_method="lexical_token_overlap",
        )
    if mode == "bm25":
        search_service = BM25SearchService(workspace)
        return RetrievalRuntime(
            search_service=search_service,
            retrieval_engine=RetrievalEngine(
                search_service,
                mode,
                "bm25",
                _snapshot_id(workspace),
            ),
            retrieval_mode=mode,
            retrieval_method="bm25",
        )

    resolved_config = config or ConfigService(workspace).load()
    semantic_search_service = SemanticSearchService(
        workspace,
        build_embedding_service(resolved_config),
    )
    if mode == "semantic":
        return RetrievalRuntime(
            search_service=semantic_search_service,
            retrieval_engine=RetrievalEngine(
                semantic_search_service,
                mode,
                "semantic_cosine_similarity",
                _snapshot_id(workspace),
            ),
            retrieval_mode=mode,
            retrieval_method="semantic_cosine_similarity",
            embedding_provider=resolved_config.embedding.provider,
            embedding_model=resolved_config.embedding.model,
            index_path=workspace.vector_index_path,
        )

    hybrid_config = HybridSearchConfig()
    hybrid_search_service = HybridSearchService(
        sparse_search_service=BM25SearchService(workspace),
        dense_search_service=semantic_search_service,
        config=hybrid_config,
    )
    return RetrievalRuntime(
        search_service=hybrid_search_service,
        retrieval_engine=RetrievalEngine(
            hybrid_search_service,
            mode,
            "hybrid_rrf",
            _snapshot_id(workspace),
        ),
        retrieval_mode=mode,
        retrieval_method="hybrid_rrf",
        embedding_provider=resolved_config.embedding.provider,
        embedding_model=resolved_config.embedding.model,
        index_path=workspace.vector_index_path,
        fusion_method="reciprocal_rank_fusion",
        rrf_k=hybrid_config.rrf_k,
        sparse_method=hybrid_config.sparse_method,
        dense_method=hybrid_config.dense_method,
        sparse_weight=hybrid_config.sparse_weight,
        dense_weight=hybrid_config.dense_weight,
        lexical_weight=hybrid_config.lexical_weight,
        semantic_weight=hybrid_config.semantic_weight,
        candidate_limit=hybrid_search_service.candidate_limit_for(limit),
    )


def _snapshot_id(workspace: ApplicationWorkspace) -> str | None:
    current_snapshot = getattr(workspace, "current_snapshot_id", None)
    if not callable(current_snapshot):
        return None
    value = current_snapshot()
    return value if isinstance(value, str) else None


def build_embedding_service(config: AppConfig) -> EmbeddingService:
    return EmbeddingService.from_config(
        config,
        http_client=default_http_client,
    )


def build_generation_service(config: AppConfig) -> GenerationService:
    return GenerationService.from_config(
        config,
        http_client=default_http_client,
    )


def build_text_generation_client(
    config: AppConfig,
) -> OpenAIResponsesTextGenerationClient:
    return OpenAIResponsesTextGenerationClient.from_config(
        config,
        http_client=default_http_client,
    )
