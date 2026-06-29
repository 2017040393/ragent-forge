from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragent_forge.app.models import DocumentChunk, IngestResult


class LocalWorkspace:
    def __init__(self, root_path: str | Path = ".ragent") -> None:
        self.root_path = Path(root_path)
        self.chunks_dir = self.root_path / "chunks"
        self.ingest_dir = self.root_path / "ingest"
        self.chunks_path = self.chunks_dir / "chunks.jsonl"
        self.latest_summary_path = self.ingest_dir / "latest_summary.json"

    def ensure_exists(self) -> None:
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.ingest_dir.mkdir(parents=True, exist_ok=True)

    def write_chunks(self, chunks: list[DocumentChunk]) -> Path:
        self.ensure_exists()
        lines = [
            json.dumps(self._chunk_record(chunk), ensure_ascii=False, sort_keys=True)
            for chunk in chunks
        ]
        content = "\n".join(lines)
        if content:
            content = f"{content}\n"
        self.chunks_path.write_text(content, encoding="utf-8")
        return self.chunks_path

    def write_ingest_summary(self, result: IngestResult) -> Path:
        self.ensure_exists()
        self.latest_summary_path.write_text(
            json.dumps(
                self._summary_record(result),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.latest_summary_path

    def _chunk_record(self, chunk: DocumentChunk) -> dict[str, Any]:
        start_char = chunk.metadata.get("start_char")
        end_char = chunk.metadata.get("end_char")
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "source_path": chunk.metadata.get("source_path"),
            "start_char": start_char,
            "end_char": end_char,
            "metadata": chunk.metadata,
        }

    def _summary_record(self, result: IngestResult) -> dict[str, Any]:
        return {
            "source_path": result.source_path,
            "document_count": result.document_count,
            "chunk_count": result.chunk_count,
            "skipped_count": result.skipped_count,
            "skipped_files": result.skipped_files,
            "metadata": result.metadata,
        }
