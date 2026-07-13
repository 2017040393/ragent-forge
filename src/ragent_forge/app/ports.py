from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ragent_forge.app.models import EmbeddingResult
from ragent_forge.core.models import DocumentChunk
from ragent_forge.core.retrieval.contracts import ChunkRecord
from ragent_forge.core.workspace import WorkspaceGenerationCommit


class ChunkReader(Protocol):
    def read_chunks(self) -> list[ChunkRecord]:
        ...


class EmbeddingServicePort(Protocol):
    provider_name: str

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        ...


class SnapshotReader(Protocol):
    def current_snapshot_id(self) -> str | None:
        ...


@runtime_checkable
class GenerationWorkspace(SnapshotReader, Protocol):
    def uses_generation_layout(self) -> bool:
        ...

    def new_snapshot_id(self) -> str:
        ...

    def commit_vector_index_generation(
        self,
        records: list[dict[str, object]],
        index_manifest: dict[str, object],
        snapshot_id: str,
    ) -> WorkspaceGenerationCommit:
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
    @property
    def chunks_path(self) -> Path:
        ...

    def write_chunks(self, chunks: list[DocumentChunk]) -> Path:
        ...


class ConfigWorkspace(Protocol):
    @property
    def root_path(self) -> Path:
        ...

    @property
    def config_path(self) -> Path:
        ...

    def atomic_write_text(self, path: str | Path, content: str) -> Path:
        ...


class SessionWorkspace(ConfigWorkspace, Protocol):
    @property
    def sessions_dir(self) -> Path:
        ...

    @property
    def session_exports_dir(self) -> Path:
        ...

    @property
    def session_index_path(self) -> Path:
        ...

    @property
    def latest_session_path(self) -> Path:
        ...

    def ensure_exists(self) -> None:
        ...

    def write_lock(self) -> AbstractContextManager[None]:
        ...


class TraceWorkspace(Protocol):
    traces_dir: Path


class VectorIndexWorkspace(SnapshotReader, Protocol):
    @property
    def index_dir(self) -> Path:
        ...

    @property
    def vector_index_path(self) -> Path:
        ...

    @property
    def vector_index_manifest_path(self) -> Path:
        ...

    def atomic_write_text(self, path: str | Path, content: str) -> Path:
        ...

    def write_lock(self) -> AbstractContextManager[None]:
        ...


class RetrievalWorkspace(
    ChunkReader,
    VectorIndexWorkspace,
    Protocol,
):
    @property
    def chunks_path(self) -> Path:
        ...


class ApplicationWorkspace(RetrievalWorkspace, ConfigWorkspace, Protocol):
    pass
