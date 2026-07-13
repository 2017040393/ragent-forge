from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ragent_forge.core.models import DocumentChunk
from ragent_forge.core.retrieval.contracts import ChunkRecord


class ChunkReader(Protocol):
    def read_chunks(self) -> list[ChunkRecord]:
        ...


class SnapshotReader(Protocol):
    def current_snapshot_id(self) -> str | None:
        ...


class HttpPostClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: Any,
        timeout: int,
    ) -> object:
        ...


class HttpStreamClient(Protocol):
    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: Any,
        timeout: int,
    ) -> Any:
        ...


@runtime_checkable
class HttpTransportErrorClassifier(Protocol):
    def is_transport_error(self, exc: Exception) -> bool:
        ...


class HttpResponse(Protocol):
    def raise_for_status(self) -> None:
        ...

    def json(self) -> object:
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


class VectorIndexWorkspace(SnapshotReader, Protocol):
    index_dir: Path
    vector_index_path: Path
    vector_index_manifest_path: Path


class RetrievalWorkspace(
    ChunkReader,
    VectorIndexWorkspace,
    Protocol,
):
    chunks_path: Path


class ApplicationWorkspace(RetrievalWorkspace, ConfigWorkspace, Protocol):
    pass
