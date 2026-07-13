from typing import Literal

from pydantic import BaseModel, Field

from ragent_forge.core.schema import WORKSPACE_SCHEMA_VERSION

GenerationArtifact = Literal[
    "chunks",
    "ingest_summary",
    "vector_index",
    "vector_index_manifest",
]


class WorkspaceSnapshotManifest(BaseModel):
    schema_version: int = WORKSPACE_SCHEMA_VERSION
    snapshot_id: str
    created_at: str
    source_path: str
    chunk_count: int
    parent_snapshot_id: str | None = None
    artifacts: list[GenerationArtifact] = Field(default_factory=list)


class WorkspaceCurrentPointer(BaseModel):
    schema_version: int = WORKSPACE_SCHEMA_VERSION
    snapshot_id: str
    committed_at: str


class WorkspaceGenerationCommit(BaseModel):
    snapshot_id: str
    generation_dir: str
    manifest_path: str
    chunks_path: str
    ingest_summary_path: str
    vector_index_path: str | None = None
    vector_index_manifest_path: str | None = None


class WorkspaceMigrationReport(BaseModel):
    dry_run: bool
    required: bool
    source_layout: Literal["empty", "legacy_flat", "generation"]
    target_schema_version: int = WORKSPACE_SCHEMA_VERSION
    snapshot_id: str | None = None
    actions: list[str] = Field(default_factory=list)
