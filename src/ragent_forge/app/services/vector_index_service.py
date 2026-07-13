from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragent_forge.app.ports import VectorIndexWorkspace
from ragent_forge.app.schema import add_schema_version, validate_schema_version
from ragent_forge.app.storage import atomic_write_text, workspace_write_lock
from ragent_forge.core.retrieval.contracts import ChunkRecord


class VectorIndexRecord(BaseModel):
    schema_version: int = 1
    snapshot_id: str | None = None
    chunk_id: str
    document_id: str
    source_path: str
    start_char: int | None = None
    end_char: int | None = None
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    embedding: list[float]
    text_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk: ChunkRecord,
        embedding_provider: str,
        embedding_model: str,
        embedding: list[float],
    ) -> VectorIndexRecord:
        text = str(chunk.get("text", ""))
        return cls(
            snapshot_id=_optional_string(chunk.get("snapshot_id")),
            chunk_id=str(chunk.get("chunk_id", "")),
            document_id=str(chunk.get("document_id", "")),
            source_path=str(chunk.get("source_path", "")),
            start_char=_optional_int(chunk.get("start_char")),
            end_char=_optional_int(chunk.get("end_char")),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dim=len(embedding),
            embedding=embedding,
            text_hash=hash_text(text),
            metadata=_metadata(chunk.get("metadata")),
        )


class VectorIndexWriteResult(BaseModel):
    index_path: Path
    manifest_path: Path
    chunk_count: int
    embedding_dim: int


class VectorIndexService:
    def __init__(self, workspace: VectorIndexWorkspace) -> None:
        self.workspace = workspace

    def write_index(
        self,
        records: list[VectorIndexRecord],
        embedding_provider: str,
        embedding_model: str,
        chunks_path: Path,
        snapshot_id: str | None = None,
    ) -> VectorIndexWriteResult:
        self.workspace.index_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(record.model_dump(), ensure_ascii=False, sort_keys=True)
            for record in records
        ]
        content = "\n".join(lines)
        if content:
            content = f"{content}\n"
        manifest = add_schema_version(
            {
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "chunk_count": len(records),
                "embedding_dim": (
                    records[0].embedding_dim if records else 0
                ),
                "built_at": _format_timestamp(datetime.now(UTC)),
                "chunks_path": str(chunks_path),
                "index_path": str(self.workspace.vector_index_path),
                "snapshot_id": snapshot_id,
            }
        )
        with workspace_write_lock():
            atomic_write_text(self.workspace.vector_index_path, content)
            atomic_write_text(
                self.workspace.vector_index_manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )

        embedding_dim = records[0].embedding_dim if records else 0
        return VectorIndexWriteResult(
            index_path=self.workspace.vector_index_path,
            manifest_path=self.workspace.vector_index_manifest_path,
            chunk_count=len(records),
            embedding_dim=embedding_dim,
        )

    def read_index(self) -> list[VectorIndexRecord]:
        if not self.workspace.vector_index_path.is_file():
            raise ValueError(
                "vector index not found. Run `ragent index build` first."
            )
        records: list[VectorIndexRecord] = []
        for line_number, line in enumerate(
            self.workspace.vector_index_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in vector index "
                    f"{self.workspace.vector_index_path} at line {line_number}: "
                    f"{exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid JSON in vector index "
                    f"{self.workspace.vector_index_path} at line {line_number}: "
                    "expected object"
                )
            validate_schema_version(payload, "vector index")
            records.append(VectorIndexRecord.model_validate(payload))
        self.read_manifest()
        active_snapshot_id = self.workspace.current_snapshot_id()
        if active_snapshot_id is not None and records:
            record_snapshot_ids = {
                record.snapshot_id for record in records if record.snapshot_id
            }
            if record_snapshot_ids != {active_snapshot_id}:
                raise ValueError(
                    "Vector index snapshot mismatch: expected "
                    f"{active_snapshot_id}, found {sorted(record_snapshot_ids)}"
                )
        return records

    def read_manifest(self) -> dict[str, Any]:
        if not self.workspace.vector_index_manifest_path.is_file():
            if self.workspace.current_snapshot_id() is not None:
                raise ValueError(
                    "Vector index manifest is missing for the active workspace "
                    "snapshot"
                )
            return {}
        try:
            manifest = json.loads(
                self.workspace.vector_index_manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON in vector index manifest "
                f"{self.workspace.vector_index_manifest_path}: {exc.msg}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ValueError(
                "Invalid JSON in vector index manifest "
                f"{self.workspace.vector_index_manifest_path}: expected object"
            )
        validate_schema_version(manifest, "vector index manifest")
        active_snapshot_id = self.workspace.current_snapshot_id()
        manifest_snapshot_id = manifest.get("snapshot_id")
        if (
            active_snapshot_id is not None
            and manifest_snapshot_id != active_snapshot_id
        ):
            raise ValueError(
                "Vector index manifest snapshot mismatch: expected "
                f"{active_snapshot_id}, found {manifest_snapshot_id}"
            )
        return manifest


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _metadata(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")
