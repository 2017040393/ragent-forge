from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragent_forge.app.ports import GenerationWorkspace, VectorIndexWorkspace
from ragent_forge.app.storage import atomic_write_text, workspace_write_lock
from ragent_forge.core.models import (
    SourceAuthority,
    SourceKind,
    SourceLifecycle,
)
from ragent_forge.core.retrieval.contracts import ChunkRecord, MetadataRecord
from ragent_forge.core.schema import (
    WORKSPACE_SCHEMA_VERSION,
    add_schema_version,
    migrate_schema_record,
)


class VectorIndexRecord(BaseModel):
    schema_version: int = WORKSPACE_SCHEMA_VERSION
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
    metadata: MetadataRecord = Field(default_factory=MetadataRecord)
    source_kind: SourceKind = "document"
    provenance: str | None = None
    authority: SourceAuthority = "source"
    freshness: str | None = None
    lifecycle: SourceLifecycle = "regenerable"

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
            source_kind=chunk.get("source_kind", "document"),
            provenance=chunk.get("provenance"),
            authority=chunk.get("authority", "source"),
            freshness=chunk.get("freshness"),
            lifecycle=chunk.get("lifecycle", "regenerable"),
        )


class VectorIndexWriteResult(BaseModel):
    index_path: Path
    manifest_path: Path
    chunk_count: int
    embedding_dim: int
    snapshot_id: str | None = None


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
        if (
            isinstance(self.workspace, GenerationWorkspace)
            and self.workspace.uses_generation_layout()
        ):
            if snapshot_id is None:
                raise ValueError(
                    "snapshot_id is required for a vector index generation commit"
                )
            commit = self.workspace.commit_vector_index_generation(
                [record.model_dump(mode="json") for record in records],
                manifest,
                snapshot_id,
            )
            if (
                commit.vector_index_path is None
                or commit.vector_index_manifest_path is None
            ):
                raise RuntimeError(
                    "Vector index generation commit did not return index paths"
                )
            return VectorIndexWriteResult(
                index_path=Path(commit.vector_index_path),
                manifest_path=Path(commit.vector_index_manifest_path),
                chunk_count=len(records),
                embedding_dim=(records[0].embedding_dim if records else 0),
                snapshot_id=commit.snapshot_id,
            )

        self.workspace.index_dir.mkdir(parents=True, exist_ok=True)
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
            snapshot_id=snapshot_id,
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
            records.append(
                VectorIndexRecord.model_validate(
                    migrate_schema_record(payload, "vector index")
                )
            )
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
        manifest = migrate_schema_record(manifest, "vector index manifest")
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


def _metadata(value: object) -> MetadataRecord:
    return MetadataRecord.from_value(value if isinstance(value, dict) else {})


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")
