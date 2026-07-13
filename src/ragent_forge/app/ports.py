from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ragent_forge.core.models import DocumentChunk


class ChunkReader(Protocol):
    def read_chunks(self) -> list[dict[str, Any]]:
        ...


class ChunkStore(ChunkReader, Protocol):
    chunks_path: Path

    def write_chunks(self, chunks: list[DocumentChunk]) -> Path:
        ...


class ConfigWorkspace(Protocol):
    root_path: Path
    config_path: Path


class TraceWorkspace(Protocol):
    traces_dir: Path


class VectorIndexWorkspace(Protocol):
    index_dir: Path
    vector_index_path: Path
    vector_index_manifest_path: Path


class RetrievalWorkspace(ChunkReader, VectorIndexWorkspace, Protocol):
    chunks_path: Path


class ApplicationWorkspace(RetrievalWorkspace, ConfigWorkspace, Protocol):
    pass
