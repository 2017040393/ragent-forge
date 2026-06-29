from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragent_forge.app.models import DocumentChunk, IngestResult, WorkspaceStatus


class LocalWorkspace:
    def __init__(self, root_path: str | Path = ".ragent") -> None:
        self.root_path = Path(root_path).expanduser()
        self.chunks_dir = self.root_path / "chunks"
        self.ingest_dir = self.root_path / "ingest"
        self.chunks_path = self.chunks_dir / "chunks.jsonl"
        self.latest_summary_path = self.ingest_dir / "latest_summary.json"

    def exists(self) -> bool:
        return self.root_path.exists()

    def has_chunks(self) -> bool:
        return self.chunks_path.is_file()

    def has_summary(self) -> bool:
        return self.latest_summary_path.is_file()

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

    def read_chunks(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.chunks_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in chunks file {self.chunks_path} "
                    f"at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid JSON in chunks file {self.chunks_path} "
                    f"at line {line_number}: expected object"
                )
            records.append(record)
        return records

    def read_ingest_summary(self) -> dict[str, Any]:
        try:
            summary = json.loads(
                self.latest_summary_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in ingest summary {self.latest_summary_path}: "
                f"{exc.msg}"
            ) from exc
        if not isinstance(summary, dict):
            raise ValueError(
                f"Invalid JSON in ingest summary {self.latest_summary_path}: "
                "expected object"
            )
        return summary

    def status(self) -> WorkspaceStatus:
        workspace_exists = self.exists()
        chunks_exist = self.has_chunks()
        summary_exists = self.has_summary()

        if not workspace_exists:
            status_name = "not_initialized"
            missing_files: list[str] = []
        else:
            missing_files = self._missing_files(chunks_exist, summary_exists)
            status_name = "incomplete" if missing_files else "ready"

        summary: dict[str, Any] = {}
        chunk_count_from_file: int | None = None
        if status_name == "ready":
            summary = self.read_ingest_summary()
            chunk_count_from_file = len(self.read_chunks())

        return WorkspaceStatus(
            root_path=str(self.root_path),
            exists=workspace_exists,
            has_chunks=chunks_exist,
            has_summary=summary_exists,
            status=status_name,
            chunks_path=str(self.chunks_path),
            latest_summary_path=str(self.latest_summary_path),
            summary=summary,
            chunk_count_from_file=chunk_count_from_file,
            missing_files=missing_files,
        )

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

    def _missing_files(self, chunks_exist: bool, summary_exists: bool) -> list[str]:
        missing_files: list[str] = []
        if not chunks_exist:
            missing_files.append(str(self.chunks_path))
        if not summary_exists:
            missing_files.append(str(self.latest_summary_path))
        return missing_files
