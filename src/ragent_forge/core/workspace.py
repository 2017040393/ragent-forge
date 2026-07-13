from pydantic import BaseModel


class WorkspaceSnapshotManifest(BaseModel):
    schema_version: int = 1
    snapshot_id: str
    created_at: str
    source_path: str
    chunk_count: int
