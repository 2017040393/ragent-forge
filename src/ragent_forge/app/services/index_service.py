from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ragent_forge.app.ports import GenerationWorkspace, RetrievalWorkspace
from ragent_forge.app.services.embedding_service import EmbeddingService
from ragent_forge.app.services.vector_index_service import (
    VectorIndexRecord,
    VectorIndexService,
)
from ragent_forge.core.retrieval.contracts import ChunkRecord


class IndexBuildResult(BaseModel):
    embedding_provider: str
    embedding_model: str
    chunk_count: int
    embedding_dim: int
    index_path: Path
    manifest_path: Path
    chunks_path: Path
    batch_size: int
    snapshot_id: str | None = None


class IndexBuildService:
    def __init__(
        self,
        workspace: RetrievalWorkspace,
        embedding_service: EmbeddingService | Any | None = None,
    ) -> None:
        self.workspace = workspace
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_index_service = VectorIndexService(workspace)

    def build(
        self,
        embedding_provider: str,
        embedding_model: str | None,
        batch_size: int,
    ) -> IndexBuildResult:
        if embedding_provider == "none":
            raise RuntimeError(
                "embedding provider is not configured. Set [embedding] "
                'provider = "openai_embeddings".'
            )
        if not embedding_model:
            raise ValueError(
                "embedding.model is required when embedding.provider is "
                "openai_embeddings"
            )
        if batch_size <= 0:
            raise ValueError("embedding.batch_size must be greater than 0")

        chunks = self.workspace.read_chunks()
        snapshot_id = self.workspace.current_snapshot_id()
        if (
            isinstance(self.workspace, GenerationWorkspace)
            and self.workspace.uses_generation_layout()
        ):
            snapshot_id = self.workspace.new_snapshot_id()
        records: list[VectorIndexRecord] = []
        for batch in _batched(chunks, batch_size):
            texts = [str(chunk.get("text", "")) for chunk in batch]
            embedding_result = self.embedding_service.embed_texts(texts)
            if len(embedding_result.embeddings) != len(batch):
                raise ValueError(
                    "embedding count does not match chunk count for index build"
                )
            paired_embeddings = zip(
                batch,
                embedding_result.embeddings,
                strict=True,
            )
            for chunk, embedding in paired_embeddings:
                index_chunk: ChunkRecord = {
                    **chunk,
                    "snapshot_id": snapshot_id,
                }
                records.append(
                    VectorIndexRecord.from_chunk(
                        chunk=index_chunk,
                        embedding_provider=embedding_result.provider_name,
                        embedding_model=embedding_result.model,
                        embedding=embedding,
                    )
                )

        write_result = self.vector_index_service.write_index(
            records,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunks_path=self.workspace.chunks_path,
            snapshot_id=snapshot_id,
        )
        return IndexBuildResult(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunk_count=write_result.chunk_count,
            embedding_dim=write_result.embedding_dim,
            index_path=write_result.index_path,
            manifest_path=write_result.manifest_path,
            chunks_path=self.workspace.chunks_path,
            batch_size=batch_size,
            snapshot_id=write_result.snapshot_id,
        )


def _batched(
    chunks: list[ChunkRecord],
    batch_size: int,
) -> list[list[ChunkRecord]]:
    return [
        chunks[start : start + batch_size]
        for start in range(0, len(chunks), batch_size)
    ]
