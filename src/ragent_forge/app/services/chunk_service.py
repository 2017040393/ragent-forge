from __future__ import annotations

import re

from ragent_forge.app.ports import ChunkReader
from ragent_forge.core.retrieval.contracts import ChunkRecord


class ChunkService:
    def __init__(self, workspace: ChunkReader) -> None:
        self.workspace = workspace

    def list_chunks(self, limit: int = 20) -> list[ChunkRecord]:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")
        return self.workspace.read_chunks()[:limit]

    def count_chunks(self) -> int:
        return len(self.workspace.read_chunks())

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        for chunk in self.workspace.read_chunks():
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None


def make_preview(text: str, max_length: int = 80) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_length:
        return normalized
    if max_length <= 3:
        return "." * max_length
    return f"{normalized[: max_length - 3]}..."
