from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ragent_forge.app.models import IngestResult, OperationTrace, RagTrace, TraceStep


class TraceService:
    def create_empty_trace(self, query: str) -> RagTrace:
        return RagTrace(query=query, metadata={"status": "stub"})


def build_ingest_trace(
    result: IngestResult,
    chunks_path: Path,
    summary_path: Path,
    started_at: datetime,
    finished_at: datetime,
) -> OperationTrace:
    started_at_utc = _as_utc(started_at)
    finished_at_utc = _as_utc(finished_at)
    metadata = {
        "source_path": result.source_path,
        "document_count": result.document_count,
        "chunk_count": result.chunk_count,
        "skipped_count": result.skipped_count,
        "chunk_size": result.metadata["chunk_size"],
        "chunk_overlap": result.metadata["chunk_overlap"],
        "chunks_path": str(chunks_path),
        "summary_path": str(summary_path),
    }
    return OperationTrace(
        trace_id=f"ingest-{started_at_utc.strftime('%Y%m%dT%H%M%SZ')}",
        operation="ingest",
        status="success",
        started_at=_format_timestamp(started_at_utc),
        finished_at=_format_timestamp(finished_at_utc),
        steps=[
            TraceStep(
                name="load_documents",
                description="Load supported Markdown/TXT documents.",
                inputs={"source_path": result.source_path},
                outputs={
                    "document_count": result.document_count,
                    "skipped_count": result.skipped_count,
                },
            ),
            TraceStep(
                name="chunk_documents",
                description="Split loaded documents into deterministic chunks.",
                inputs={
                    "document_count": result.document_count,
                    "chunk_size": result.metadata["chunk_size"],
                    "chunk_overlap": result.metadata["chunk_overlap"],
                },
                outputs={"chunk_count": result.chunk_count},
            ),
            TraceStep(
                name="write_chunks",
                description="Persist chunk records to the local workspace.",
                inputs={"chunk_count": result.chunk_count},
                outputs={"chunks_path": str(chunks_path)},
            ),
            TraceStep(
                name="write_ingest_summary",
                description="Persist the latest ingestion summary.",
                inputs={
                    "document_count": result.document_count,
                    "chunk_count": result.chunk_count,
                    "skipped_count": result.skipped_count,
                },
                outputs={"summary_path": str(summary_path)},
            ),
        ],
        metadata=metadata,
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")
