from __future__ import annotations

from datetime import UTC, datetime

from rich.console import Console

from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.index_service import IndexBuildService
from ragent_forge.app.services.trace_service import build_index_build_trace
from ragent_forge.app.services.vector_index_service import VectorIndexService
from ragent_forge.cli.handlers.chunks import _print_no_chunks
from ragent_forge.composition import build_embedding_service
from ragent_forge.core.retrieval.representations import EmbeddingRepresentation
from ragent_forge.infrastructure.local_workspace import LocalWorkspace


def _handle_index_build(
    console: Console,
    workspace_path: str,
    *,
    embedding_representation: EmbeddingRepresentation = "raw_chunk_text_v1",
) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        _print_no_chunks(console)
        return 0
    started_at = datetime.now(UTC)
    try:
        config = ConfigService(workspace).load()
        embedding_service = build_embedding_service(config)
        result = IndexBuildService(
            workspace, embedding_service=embedding_service
        ).build(
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model,
            batch_size=config.embedding.batch_size,
            embedding_representation=embedding_representation,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"Index build failed: {exc}", markup=False, soft_wrap=True)
        return 1
    finished_at = datetime.now(UTC)
    trace = build_index_build_trace(
        embedding_provider=result.embedding_provider,
        embedding_model=result.embedding_model,
        chunk_count=result.chunk_count,
        embedding_dim=result.embedding_dim,
        index_path=result.index_path,
        chunks_path=result.chunks_path,
        batch_size=result.batch_size,
        started_at=started_at,
        finished_at=finished_at,
        snapshot_id=result.snapshot_id,
        embedding_representation=result.embedding_representation,
    )
    trace_path = workspace.write_trace(trace)
    console.print("Semantic index build")
    console.print()
    console.print(f"Embedding provider: {result.embedding_provider}")
    console.print(f"Embedding model: {result.embedding_model}")
    console.print(f"Embedding representation: {result.embedding_representation}")
    console.print(f"Chunks embedded: {result.chunk_count}")
    console.print(f"Embedding dim: {result.embedding_dim}")
    console.print(f"Index path: {result.index_path}")
    console.print(f"Saved trace to: {trace_path}")
    return 0


def _handle_index_status(console: Console, workspace_path: str) -> int:
    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_vector_index():
        console.print("Semantic index: missing")
        console.print("Run `ragent index build` to create it.")
        return 0
    try:
        manifest = VectorIndexService(workspace).read_manifest()
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Index status failed:[/bold red] {exc}")
        return 1
    console.print("Semantic index: ready")
    index_path = manifest.get("index_path", workspace.vector_index_path)
    console.print(f"Index path: {index_path}")
    console.print(f"Chunks indexed: {manifest.get('chunk_count', 0)}")
    console.print(f"Embedding model: {manifest.get('embedding_model', '')}")
    console.print(f"Embedding dim: {manifest.get('embedding_dim', 0)}")
    console.print(
        "Embedding representation: "
        f"{manifest.get('embedding_representation', 'unknown')}"
    )
    console.print(f"Built at: {manifest.get('built_at', '')}")
    return 0
