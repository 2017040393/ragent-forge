from __future__ import annotations

import json
import shutil
import uuid
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragent_forge.app.models import (
    DocumentChunk,
    IngestResult,
    OperationTrace,
    WorkspaceStatus,
)
from ragent_forge.core.models import (
    SourceAuthority,
    SourceKind,
    SourceLifecycle,
    is_source_authority,
    is_source_kind,
    is_source_lifecycle,
)
from ragent_forge.core.retrieval.contracts import ChunkRecord
from ragent_forge.core.schema import (
    WORKSPACE_SCHEMA_VERSION,
    add_schema_version,
    migrate_schema_record,
)
from ragent_forge.core.workspace import (
    GenerationArtifact,
    WorkspaceCurrentPointer,
    WorkspaceGenerationCommit,
    WorkspaceMigrationReport,
    WorkspaceSnapshotManifest,
)
from ragent_forge.infrastructure.storage import (
    atomic_write_text,
    workspace_write_lock,
)


class LocalWorkspace:
    def __init__(self, root_path: str | Path = ".ragent") -> None:
        self.root_path = Path(root_path).expanduser()
        self.generations_dir = self.root_path / "generations"
        self.current_path = self.root_path / "current.json"
        self._legacy_chunks_dir = self.root_path / "chunks"
        self._legacy_ingest_dir = self.root_path / "ingest"
        self._legacy_index_dir = self.root_path / "index"
        self._legacy_chunks_path = self._legacy_chunks_dir / "chunks.jsonl"
        self._legacy_summary_path = (
            self._legacy_ingest_dir / "latest_summary.json"
        )
        self._legacy_snapshot_manifest_path = self.root_path / "snapshot.json"
        self._legacy_vector_index_path = (
            self._legacy_index_dir / "vector_index.jsonl"
        )
        self._legacy_vector_index_manifest_path = (
            self._legacy_index_dir / "vector_index_manifest.json"
        )
        self.traces_dir = self.root_path / "traces"
        self.eval_dir = self.root_path / "eval"
        self.eval_runs_dir = self.eval_dir / "runs"
        self.sessions_dir = self.root_path / "sessions"
        self.session_exports_dir = self.sessions_dir / "exports"
        self.latest_trace_path = self.traces_dir / "latest_trace.json"
        self.latest_retrieval_eval_path = (
            self.eval_dir / "latest_retrieval_eval.json"
        )
        self.latest_retrieval_compare_path = (
            self.eval_dir / "latest_retrieval_compare.json"
        )
        self.session_index_path = self.sessions_dir / "index.json"
        self.latest_session_path = self.sessions_dir / "latest.json"
        self.config_path = self.root_path / "config.toml"

    @property
    def chunks_dir(self) -> Path:
        return self.chunks_path.parent

    @property
    def ingest_dir(self) -> Path:
        return self.latest_summary_path.parent

    @property
    def index_dir(self) -> Path:
        return self.vector_index_path.parent

    @property
    def chunks_path(self) -> Path:
        return self._generation_artifact_path(
            "chunks.jsonl",
            self._legacy_chunks_path,
        )

    @property
    def latest_summary_path(self) -> Path:
        return self._generation_artifact_path(
            "ingest_summary.json",
            self._legacy_summary_path,
        )

    @property
    def snapshot_manifest_path(self) -> Path:
        return self._generation_artifact_path(
            "manifest.json",
            self._legacy_snapshot_manifest_path,
        )

    @property
    def vector_index_path(self) -> Path:
        return self._generation_artifact_path(
            "vector_index.jsonl",
            self._legacy_vector_index_path,
        )

    @property
    def vector_index_manifest_path(self) -> Path:
        return self._generation_artifact_path(
            "vector_index_manifest.json",
            self._legacy_vector_index_manifest_path,
        )

    def exists(self) -> bool:
        return self.root_path.exists()

    def has_chunks(self) -> bool:
        return self.chunks_path.is_file()

    def has_summary(self) -> bool:
        return self.latest_summary_path.is_file()

    def has_latest_trace(self) -> bool:
        return self.latest_trace_path.is_file()

    def has_vector_index(self) -> bool:
        return self.vector_index_path.is_file()

    def ensure_exists(self) -> None:
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.eval_runs_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_exports_dir.mkdir(parents=True, exist_ok=True)

    def uses_generation_layout(self) -> bool:
        return self.current_path.is_file()

    def atomic_write_text(self, path: str | Path, content: str) -> Path:
        return atomic_write_text(path, content)

    def write_lock(self) -> AbstractContextManager[None]:
        return workspace_write_lock()

    def new_snapshot_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"snapshot-{timestamp}-{uuid.uuid4().hex[:8]}"

    def current_snapshot_id(self) -> str | None:
        if self.current_path.is_file():
            pointer = self._read_current_pointer()
            manifest = self._read_generation_manifest(pointer.snapshot_id)
            if manifest.snapshot_id != pointer.snapshot_id:
                raise ValueError(
                    "Workspace current pointer mismatch: expected "
                    f"{pointer.snapshot_id}, found {manifest.snapshot_id}"
                )
            return pointer.snapshot_id
        manifest = self._read_legacy_snapshot_manifest()
        return manifest.snapshot_id if manifest is not None else None

    def read_snapshot_manifest(self) -> WorkspaceSnapshotManifest | None:
        if self.current_path.is_file():
            return self._read_generation_manifest(
                self._read_current_pointer().snapshot_id
            )
        return self._read_legacy_snapshot_manifest()

    def commit_ingest_generation(
        self,
        result: IngestResult,
        snapshot_id: str | None = None,
    ) -> WorkspaceGenerationCommit:
        resolved_snapshot_id = snapshot_id or self.new_snapshot_id()
        chunks_content = self._chunks_content(
            result.chunks,
            resolved_snapshot_id,
        )
        summary_content = self._json_content(
            self._summary_record(result, resolved_snapshot_id)
        )
        manifest = WorkspaceSnapshotManifest(
            snapshot_id=resolved_snapshot_id,
            created_at=_format_timestamp(datetime.now(UTC)),
            source_path=result.source_path,
            chunk_count=result.chunk_count,
            artifacts=["chunks", "ingest_summary"],
        )
        return self._commit_generation(
            manifest,
            {
                "chunks.jsonl": chunks_content,
                "ingest_summary.json": summary_content,
            },
        )

    def commit_vector_index_generation(
        self,
        records: list[dict[str, object]],
        index_manifest: dict[str, object],
        snapshot_id: str,
    ) -> WorkspaceGenerationCommit:
        if not self.current_path.is_file():
            raise RuntimeError(
                "Vector index generation commits require an active generation"
            )
        parent_manifest = self.read_snapshot_manifest()
        if parent_manifest is None:
            raise RuntimeError("Active workspace generation manifest is missing")

        chunks = [
            add_schema_version({**chunk, "snapshot_id": snapshot_id})
            for chunk in self.read_chunks()
        ]
        summary = add_schema_version(
            {**self.read_ingest_summary(), "snapshot_id": snapshot_id}
        )
        generation_dir = self.generations_dir / snapshot_id
        normalized_records = [
            add_schema_version({**record, "snapshot_id": snapshot_id})
            for record in records
        ]
        normalized_index_manifest = add_schema_version(
            {
                **index_manifest,
                "snapshot_id": snapshot_id,
                "chunks_path": str(generation_dir / "chunks.jsonl"),
                "index_path": str(generation_dir / "vector_index.jsonl"),
            }
        )
        manifest = WorkspaceSnapshotManifest(
            snapshot_id=snapshot_id,
            created_at=_format_timestamp(datetime.now(UTC)),
            source_path=parent_manifest.source_path,
            chunk_count=len(chunks),
            parent_snapshot_id=parent_manifest.snapshot_id,
            artifacts=[
                "chunks",
                "ingest_summary",
                "vector_index",
                "vector_index_manifest",
            ],
        )
        return self._commit_generation(
            manifest,
            {
                "chunks.jsonl": self._jsonl_content(chunks),
                "ingest_summary.json": self._json_content(summary),
                "vector_index.jsonl": self._jsonl_content(normalized_records),
                "vector_index_manifest.json": self._json_content(
                    normalized_index_manifest
                ),
            },
        )

    def migrate_legacy_workspace(
        self,
        *,
        dry_run: bool = False,
    ) -> WorkspaceMigrationReport:
        if self.current_path.is_file():
            snapshot_id = self.current_snapshot_id()
            return WorkspaceMigrationReport(
                dry_run=dry_run,
                required=False,
                source_layout="generation",
                snapshot_id=snapshot_id,
                actions=["No migration required; current.json is already active."],
            )
        if (
            not self._legacy_chunks_path.exists()
            and not self._legacy_summary_path.exists()
        ):
            return WorkspaceMigrationReport(
                dry_run=dry_run,
                required=False,
                source_layout="empty",
                actions=["No legacy generation artifacts were found."],
            )
        missing = [
            str(path)
            for path in (self._legacy_chunks_path, self._legacy_summary_path)
            if not path.is_file()
        ]
        if missing:
            raise ValueError(
                "Cannot migrate incomplete legacy workspace; missing: "
                + ", ".join(missing)
            )

        legacy_manifest = self._read_legacy_snapshot_manifest()
        snapshot_id = (
            legacy_manifest.snapshot_id
            if legacy_manifest is not None
            else self.new_snapshot_id()
        )
        chunks = self._read_chunk_records(self._legacy_chunks_path)
        chunks = [
            add_schema_version({**chunk, "snapshot_id": snapshot_id})
            for chunk in chunks
        ]
        summary = self._read_json_object(
            self._legacy_summary_path,
            "ingest summary",
        )
        summary = add_schema_version({**summary, "snapshot_id": snapshot_id})
        source_path = str(summary.get("source_path", ""))
        artifacts: dict[str, str] = {
            "chunks.jsonl": self._jsonl_content(chunks),
            "ingest_summary.json": self._json_content(summary),
        }
        manifest_artifacts: list[GenerationArtifact] = [
            "chunks",
            "ingest_summary",
        ]

        has_index = self._legacy_vector_index_path.is_file()
        has_index_manifest = self._legacy_vector_index_manifest_path.is_file()
        if has_index != has_index_manifest:
            raise ValueError(
                "Cannot migrate incomplete legacy vector index; both index and "
                "manifest are required"
            )
        if has_index:
            index_records = self._read_jsonl_objects(
                self._legacy_vector_index_path,
                "vector index",
            )
            index_records = [
                add_schema_version({**record, "snapshot_id": snapshot_id})
                for record in index_records
            ]
            index_manifest = self._read_json_object(
                self._legacy_vector_index_manifest_path,
                "vector index manifest",
            )
            generation_dir = self.generations_dir / snapshot_id
            index_manifest = add_schema_version(
                {
                    **index_manifest,
                    "snapshot_id": snapshot_id,
                    "chunks_path": str(generation_dir / "chunks.jsonl"),
                    "index_path": str(generation_dir / "vector_index.jsonl"),
                }
            )
            artifacts["vector_index.jsonl"] = self._jsonl_content(index_records)
            artifacts["vector_index_manifest.json"] = self._json_content(
                index_manifest
            )
            manifest_artifacts.extend(
                ["vector_index", "vector_index_manifest"]
            )

        actions = [
            f"Create generations/{snapshot_id} with schema version "
            f"{WORKSPACE_SCHEMA_VERSION}.",
            "Validate all generation artifacts before publication.",
            "Atomically replace current.json after generation publication.",
            "Preserve legacy flat files for rollback and inspection.",
        ]
        if not dry_run:
            manifest = WorkspaceSnapshotManifest(
                snapshot_id=snapshot_id,
                created_at=(
                    legacy_manifest.created_at
                    if legacy_manifest is not None
                    else _format_timestamp(datetime.now(UTC))
                ),
                source_path=(
                    legacy_manifest.source_path
                    if legacy_manifest is not None
                    else source_path
                ),
                chunk_count=len(chunks),
                artifacts=manifest_artifacts,
            )
            self._commit_generation(manifest, artifacts)

        return WorkspaceMigrationReport(
            dry_run=dry_run,
            required=True,
            source_layout="legacy_flat",
            snapshot_id=snapshot_id,
            actions=actions,
        )

    def commit_snapshot(
        self,
        snapshot_id: str,
        source_path: str,
        chunk_count: int,
    ) -> Path:
        self.ensure_exists()
        manifest = WorkspaceSnapshotManifest(
            snapshot_id=snapshot_id,
            created_at=_format_timestamp(datetime.now(UTC)),
            source_path=source_path,
            chunk_count=chunk_count,
        )
        if self.current_path.is_file():
            raise RuntimeError(
                "Cannot mutate an active generation; commit a new generation instead"
            )
        atomic_write_text(
            self._legacy_snapshot_manifest_path,
            json.dumps(
                add_schema_version(manifest.model_dump()),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        return self._legacy_snapshot_manifest_path

    def write_chunks(
        self,
        chunks: list[DocumentChunk],
        snapshot_id: str | None = None,
    ) -> Path:
        self.ensure_exists()
        if self.current_path.is_file():
            raise RuntimeError(
                "Cannot mutate an active generation; commit a new generation instead"
            )
        content = self._chunks_content(chunks, snapshot_id)
        atomic_write_text(self._legacy_chunks_path, content)
        return self._legacy_chunks_path

    def write_ingest_summary(
        self,
        result: IngestResult,
        snapshot_id: str | None = None,
    ) -> Path:
        self.ensure_exists()
        if self.current_path.is_file():
            raise RuntimeError(
                "Cannot mutate an active generation; commit a new generation instead"
            )
        atomic_write_text(
            self._legacy_summary_path,
            self._json_content(self._summary_record(result, snapshot_id)),
        )
        return self._legacy_summary_path

    def write_trace(
        self,
        trace: OperationTrace,
        snapshot_id: str | None = None,
    ) -> Path:
        self.ensure_exists()
        resolved_snapshot_id = snapshot_id or self.current_snapshot_id()
        if resolved_snapshot_id is not None:
            trace = trace.model_copy(
                update={
                    "metadata": {
                        **trace.metadata,
                        "snapshot_id": resolved_snapshot_id,
                    }
                }
            )
        content = (
            json.dumps(
                add_schema_version(trace.model_dump(exclude_none=True)),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        trace_path = self.traces_dir / f"{trace.trace_id}.json"
        with workspace_write_lock():
            atomic_write_text(trace_path, content)
            atomic_write_text(self.latest_trace_path, content)
        return self.latest_trace_path

    def write_retrieval_eval_report(
        self,
        report: dict[str, Any],
        report_path: str | Path | None = None,
    ) -> Path:
        self.ensure_exists()
        content = json.dumps(
            add_schema_version(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        with workspace_write_lock():
            destination = (
                self._next_eval_json_path("retrieval_eval")
                if report_path is None
                else Path(report_path).expanduser()
            )
            atomic_write_text(destination, content)
            atomic_write_text(self.latest_retrieval_eval_path, content)
        return destination

    def write_retrieval_compare_report(
        self,
        report: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> Path:
        self.ensure_exists()
        content = json.dumps(
            add_schema_version(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        with workspace_write_lock():
            destination = (
                self._next_eval_json_path("retrieval_compare")
                if output_path is None
                else Path(output_path).expanduser()
            )
            atomic_write_text(destination, content)
            atomic_write_text(self.latest_retrieval_compare_path, content)
        return destination

    def write_retrieval_eval_run(
        self,
        report: dict[str, Any],
        report_path: str | Path | None = None,
    ) -> Path:
        self.ensure_exists()
        with workspace_write_lock():
            run_dir = self._next_retrieval_eval_run_dir()
            run_dir.mkdir(parents=True)

        summary_json_path = run_dir / "summary.json"
        summary_md_path = run_dir / "summary.md"
        cases_jsonl_path = run_dir / "cases.jsonl"
        failures_jsonl_path = run_dir / "failures.jsonl"

        atomic_write_text(
            summary_json_path,
            json.dumps(
                add_schema_version(report),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        atomic_write_text(
            summary_md_path,
            self._retrieval_eval_summary_markdown(
                report,
                summary_json_path=summary_json_path,
                cases_jsonl_path=cases_jsonl_path,
                failures_jsonl_path=failures_jsonl_path,
                report_path=Path(report_path) if report_path is not None else None,
            ),
        )

        compact_cases = [
            self._compact_retrieval_eval_case(result)
            for result in _list_of_dicts(report.get("results"))
        ]
        self._write_jsonl_records(cases_jsonl_path, compact_cases)
        self._write_jsonl_records(
            failures_jsonl_path,
            [record for record in compact_cases if not record["passed"]],
        )
        return run_dir

    def read_chunks(self) -> list[ChunkRecord]:
        records = self._read_chunk_records(self.chunks_path)
        self._validate_active_snapshot(
            [record.get("snapshot_id") for record in records]
        )
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
        summary = migrate_schema_record(summary, "ingest summary")
        self._validate_active_snapshot([summary.get("snapshot_id")])
        return summary

    def read_latest_trace(self) -> dict[str, Any]:
        try:
            trace = json.loads(self.latest_trace_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in latest trace {self.latest_trace_path}: {exc.msg}"
            ) from exc
        if not isinstance(trace, dict):
            raise ValueError(
                f"Invalid JSON in latest trace {self.latest_trace_path}: "
                "expected object"
            )
        return migrate_schema_record(trace, "latest trace")

    def _validate_active_snapshot(self, snapshot_ids: list[object]) -> None:
        active_snapshot_id = self.current_snapshot_id()
        if not snapshot_ids:
            return
        distinct_snapshot_ids = {
            snapshot_id for snapshot_id in snapshot_ids if snapshot_id is not None
        }
        if active_snapshot_id is None:
            if distinct_snapshot_ids:
                raise ValueError(
                    "Workspace snapshot manifest is missing for snapshot-tagged "
                    "artifacts: "
                    f"{sorted(str(value) for value in distinct_snapshot_ids)}"
                )
            return
        if distinct_snapshot_ids != {active_snapshot_id}:
            raise ValueError(
                "Workspace snapshot mismatch: expected "
                f"{active_snapshot_id}, found "
                f"{sorted(str(value) for value in distinct_snapshot_ids)}"
            )

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
            snapshot_id=self.current_snapshot_id(),
        )

    def _chunk_record(
        self,
        chunk: DocumentChunk,
        snapshot_id: str | None,
    ) -> dict[str, Any]:
        start_char = chunk.metadata.get("start_char")
        end_char = chunk.metadata.get("end_char")
        record = add_schema_version({
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "source_path": chunk.metadata.get("source_path"),
            "start_char": start_char,
            "end_char": end_char,
            "metadata": chunk.metadata,
            "source_kind": _source_kind(chunk.metadata.get("source_kind")),
            "provenance": _optional_string(chunk.metadata.get("provenance")),
            "authority": _source_authority(chunk.metadata.get("authority")),
            "freshness": _optional_string(chunk.metadata.get("freshness")),
            "lifecycle": _source_lifecycle(chunk.metadata.get("lifecycle")),
        })
        if snapshot_id is not None:
            record["snapshot_id"] = snapshot_id
        return record

    def _summary_record(
        self,
        result: IngestResult,
        snapshot_id: str | None,
    ) -> dict[str, Any]:
        record = add_schema_version({
            "source_path": result.source_path,
            "document_count": result.document_count,
            "chunk_count": result.chunk_count,
            "skipped_count": result.skipped_count,
            "skipped_files": result.skipped_files,
            "metadata": result.metadata,
        })
        if snapshot_id is not None:
            record["snapshot_id"] = snapshot_id
        return record

    def _missing_files(self, chunks_exist: bool, summary_exists: bool) -> list[str]:
        missing_files: list[str] = []
        if not chunks_exist:
            missing_files.append(str(self.chunks_path))
        if not summary_exists:
            missing_files.append(str(self.latest_summary_path))
        return missing_files

    def _next_retrieval_eval_run_dir(self) -> Path:
        base_name = f"retrieval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        candidate = self.eval_runs_dir / base_name
        index = 1
        while candidate.exists():
            candidate = self.eval_runs_dir / f"{base_name}-{index:03d}"
            index += 1
        return candidate

    def _next_eval_json_path(self, prefix: str) -> Path:
        base_name = f"{prefix}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        candidate = self.eval_dir / f"{base_name}.json"
        index = 1
        while candidate.exists():
            candidate = self.eval_dir / f"{base_name}-{index:03d}.json"
            index += 1
        return candidate

    def _retrieval_eval_summary_markdown(
        self,
        report: dict[str, Any],
        *,
        summary_json_path: Path,
        cases_jsonl_path: Path,
        failures_jsonl_path: Path,
        report_path: Path | None,
    ) -> str:
        metrics = _dict_value(report.get("metrics"))
        failed_count = _int_value(report.get("failed_count"))
        lines = [
            "# Retrieval Eval Run",
            "",
            f"- Retrieval mode: {report.get('retrieval_mode', '')}",
            f"- Retrieval method: {report.get('retrieval_method', '')}",
            f"- Cases: {_int_value(report.get('case_count'))}",
            f"- Passed: {_int_value(report.get('passed_count'))}",
            f"- Failed: {failed_count}",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, int | float):
                lines.append(f"| {key} | {float(value):.4f} |")
            else:
                lines.append(f"| {key} | {value} |")

        lines.extend(
            [
                "",
                "## Failure Breakdown",
                "",
            ]
        )
        failure_breakdown = _failure_breakdown(report)
        if failure_breakdown:
            lines.extend(
                [
                    "| Failure Type | Count |",
                    "| --- | ---: |",
                ]
            )
            for failure_type, count in sorted(failure_breakdown.items()):
                lines.append(f"| {failure_type} | {count} |")
        else:
            lines.append("No failed cases.")

        lines.extend(
            [
                "",
                "## Report Paths",
                "",
                f"- summary.json: {summary_json_path}",
                f"- cases.jsonl: {cases_jsonl_path}",
                f"- failures.jsonl: {failures_jsonl_path}",
            ]
        )
        if report_path is not None:
            lines.append(f"- compatibility report: {report_path}")
        lines.extend(
            [
                "",
                f"Top failure count: {min(failed_count, 5)}",
                "",
            ]
        )
        return "\n".join(lines)

    def _compact_retrieval_eval_case(self, result: dict[str, Any]) -> dict[str, Any]:
        metadata = _dict_value(result.get("metadata"))
        per_case_metric_keys = (
            "retrieved_count",
            "expected_chunk_count",
            "relevant_retrieved_count",
            "relevant_result_ranks",
            "recall",
            "precision",
            "ndcg",
            "evidence_coverage",
            "mapping_coverage",
            "context_evidence_density",
            "duplicate_context_ratio",
            "retrieval_latency_ms",
            "retrieved_context_chars",
            "estimated_context_tokens",
        )
        compact_metadata = dict(metadata)
        for key in per_case_metric_keys:
            if key in result:
                compact_metadata[key] = result[key]
        return {
            "id": str(result.get("id", "")),
            "query": str(result.get("query", "")),
            "passed": bool(result.get("passed", False)),
            "failure_type": _optional_string(result.get("failure_type")),
            "failure_reason": _optional_string(result.get("failure_reason")),
            "rank": result.get("rank"),
            "matched_by": str(result.get("matched_by", "")),
            "expected_chunk_ids": _string_list(result.get("expected_chunk_ids")),
            "expected_source_paths": _string_list(result.get("expected_source_paths")),
            "actual_chunk_ids": _string_list(result.get("actual_chunk_ids")),
            "actual_source_paths": _string_list(result.get("actual_source_paths")),
            "metadata": compact_metadata,
        }

    def _write_jsonl_records(
        self,
        path: Path,
        records: list[dict[str, Any]],
    ) -> None:
        lines = [
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in records
        ]
        content = "\n".join(lines)
        if content:
            content = f"{content}\n"
        atomic_write_text(path, content)

    def _generation_artifact_path(
        self,
        filename: str,
        legacy_path: Path,
    ) -> Path:
        if not self.current_path.is_file():
            return legacy_path
        pointer = self._read_current_pointer()
        return self.generations_dir / pointer.snapshot_id / filename

    def _read_current_pointer(self) -> WorkspaceCurrentPointer:
        payload = self._read_json_object(
            self.current_path,
            "workspace current pointer",
        )
        return WorkspaceCurrentPointer.model_validate(payload)

    def _read_legacy_snapshot_manifest(
        self,
    ) -> WorkspaceSnapshotManifest | None:
        if not self._legacy_snapshot_manifest_path.is_file():
            return None
        payload = self._read_json_object(
            self._legacy_snapshot_manifest_path,
            "snapshot manifest",
        )
        return WorkspaceSnapshotManifest.model_validate(payload)

    def _read_generation_manifest(
        self,
        snapshot_id: str,
    ) -> WorkspaceSnapshotManifest:
        manifest_path = self.generations_dir / snapshot_id / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Workspace generation manifest is missing for current snapshot "
                f"{snapshot_id}: {manifest_path}"
            )
        payload = self._read_json_object(manifest_path, "generation manifest")
        manifest = WorkspaceSnapshotManifest.model_validate(payload)
        if manifest.snapshot_id != snapshot_id:
            raise ValueError(
                "Workspace generation directory mismatch: expected "
                f"{snapshot_id}, found {manifest.snapshot_id}"
            )
        return manifest

    def _commit_generation(
        self,
        manifest: WorkspaceSnapshotManifest,
        artifacts: dict[str, str],
    ) -> WorkspaceGenerationCommit:
        snapshot_id = manifest.snapshot_id
        if (
            not snapshot_id
            or Path(snapshot_id).name != snapshot_id
            or snapshot_id in {".", ".."}
        ):
            raise ValueError(f"Invalid workspace snapshot id: {snapshot_id!r}")
        required_files = {"chunks.jsonl", "ingest_summary.json"}
        missing_files = required_files.difference(artifacts)
        if missing_files:
            raise ValueError(
                "Generation commit is missing required artifacts: "
                + ", ".join(sorted(missing_files))
            )
        unknown_files = {
            name
            for name in artifacts
            if Path(name).name != name or name == "manifest.json"
        }
        if unknown_files:
            raise ValueError(
                "Generation commit contains invalid artifact names: "
                + ", ".join(sorted(unknown_files))
            )

        self._ensure_runtime_dirs()
        final_dir = self.generations_dir / snapshot_id
        temporary_dir = self.generations_dir / (
            f".{snapshot_id}.{uuid.uuid4().hex}.tmp"
        )
        pointer = WorkspaceCurrentPointer(
            snapshot_id=snapshot_id,
            committed_at=_format_timestamp(datetime.now(UTC)),
        )
        with workspace_write_lock():
            if final_dir.exists():
                raise FileExistsError(
                    f"Workspace generation already exists: {final_dir}"
                )
            temporary_dir.mkdir(parents=False)
            try:
                for filename, content in artifacts.items():
                    atomic_write_text(temporary_dir / filename, content)
                atomic_write_text(
                    temporary_dir / "manifest.json",
                    self._json_content(manifest.model_dump()),
                )
                self._validate_generation_directory(temporary_dir, manifest)
                _publish_generation_directory(temporary_dir, final_dir)
                atomic_write_text(
                    self.current_path,
                    self._json_content(pointer.model_dump()),
                )
            finally:
                if temporary_dir.exists():
                    shutil.rmtree(temporary_dir)

        vector_index_path = final_dir / "vector_index.jsonl"
        vector_manifest_path = final_dir / "vector_index_manifest.json"
        return WorkspaceGenerationCommit(
            snapshot_id=snapshot_id,
            generation_dir=str(final_dir),
            manifest_path=str(final_dir / "manifest.json"),
            chunks_path=str(final_dir / "chunks.jsonl"),
            ingest_summary_path=str(final_dir / "ingest_summary.json"),
            vector_index_path=(
                str(vector_index_path) if vector_index_path.is_file() else None
            ),
            vector_index_manifest_path=(
                str(vector_manifest_path)
                if vector_manifest_path.is_file()
                else None
            ),
        )

    def _validate_generation_directory(
        self,
        generation_dir: Path,
        expected_manifest: WorkspaceSnapshotManifest,
    ) -> None:
        manifest_payload = self._read_json_object(
            generation_dir / "manifest.json",
            "generation manifest",
        )
        manifest = WorkspaceSnapshotManifest.model_validate(manifest_payload)
        if manifest != expected_manifest:
            raise ValueError("Generation manifest changed during commit validation")
        chunks = self._read_chunk_records(generation_dir / "chunks.jsonl")
        if len(chunks) != manifest.chunk_count:
            raise ValueError(
                "Generation chunk count mismatch: manifest declares "
                f"{manifest.chunk_count}, found {len(chunks)}"
            )
        self._validate_snapshot_ids(
            manifest.snapshot_id,
            [record.get("snapshot_id") for record in chunks],
            "generation chunks",
        )
        summary = self._read_json_object(
            generation_dir / "ingest_summary.json",
            "ingest summary",
        )
        self._validate_snapshot_ids(
            manifest.snapshot_id,
            [summary.get("snapshot_id")],
            "generation ingest summary",
        )
        index_path = generation_dir / "vector_index.jsonl"
        index_manifest_path = generation_dir / "vector_index_manifest.json"
        if index_path.is_file() != index_manifest_path.is_file():
            raise ValueError(
                "Generation vector index and manifest must be committed together"
            )
        if index_path.is_file():
            index_records = self._read_jsonl_objects(index_path, "vector index")
            self._validate_snapshot_ids(
                manifest.snapshot_id,
                [record.get("snapshot_id") for record in index_records],
                "generation vector index",
            )
            index_manifest = self._read_json_object(
                index_manifest_path,
                "vector index manifest",
            )
            self._validate_snapshot_ids(
                manifest.snapshot_id,
                [index_manifest.get("snapshot_id")],
                "generation vector index manifest",
            )

    def _ensure_runtime_dirs(self) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.eval_runs_dir.mkdir(parents=True, exist_ok=True)
        self.session_exports_dir.mkdir(parents=True, exist_ok=True)

    def _chunks_content(
        self,
        chunks: list[DocumentChunk],
        snapshot_id: str | None,
    ) -> str:
        return self._jsonl_content(
            [self._chunk_record(chunk, snapshot_id) for chunk in chunks]
        )

    def _read_chunk_records(self, path: Path) -> list[ChunkRecord]:
        return [
            _normalize_chunk_record(record)
            for record in self._read_jsonl_objects(path, "chunks file")
        ]

    def _read_jsonl_objects(
        self,
        path: Path,
        artifact: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {artifact} {path} at line "
                    f"{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid JSON in {artifact} {path} at line "
                    f"{line_number}: expected object"
                )
            records.append(migrate_schema_record(record, artifact))
        return records

    def _read_json_object(self, path: Path, artifact: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {artifact} {path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid {artifact} {path}: expected object")
        return migrate_schema_record(payload, artifact)

    def _json_content(self, record: dict[str, Any]) -> str:
        return (
            json.dumps(
                add_schema_version(record),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _jsonl_content(self, records: list[dict[str, Any]]) -> str:
        lines = [
            json.dumps(
                add_schema_version(record),
                ensure_ascii=False,
                sort_keys=True,
            )
            for record in records
        ]
        return "" if not lines else "\n".join(lines) + "\n"

    def _validate_snapshot_ids(
        self,
        expected_snapshot_id: str,
        values: list[object],
        artifact: str,
    ) -> None:
        if not values:
            return
        actual = {value for value in values if isinstance(value, str) and value}
        if actual != {expected_snapshot_id}:
            raise ValueError(
                f"{artifact} snapshot mismatch: expected {expected_snapshot_id}, "
                f"found {sorted(actual)}"
            )


def _dict_value(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {str(key): item for key, item in item.items()}
        for item in value
        if isinstance(item, dict)
    ]


def _failure_breakdown(report: dict[str, Any]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for result in _list_of_dicts(report.get("results")):
        if bool(result.get("passed", False)):
            continue
        failure_type = _optional_string(result.get("failure_type")) or "unknown"
        breakdown[failure_type] = breakdown.get(failure_type, 0) + 1
    return breakdown


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _normalize_chunk_record(record: dict[str, Any]) -> ChunkRecord:
    source_path = record.get("source_path")
    snapshot_id = record.get("snapshot_id")
    metadata = record.get("metadata")
    schema_version = record.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        schema_version = WORKSPACE_SCHEMA_VERSION
    return {
        "schema_version": schema_version,
        "snapshot_id": snapshot_id if isinstance(snapshot_id, str) else None,
        "chunk_id": str(record.get("chunk_id", "")),
        "document_id": str(record.get("document_id", "")),
        "text": str(record.get("text", "")),
        "source_path": source_path if isinstance(source_path, str) else None,
        "start_char": _optional_int(record.get("start_char")),
        "end_char": _optional_int(record.get("end_char")),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "source_kind": _source_kind(record.get("source_kind")),
        "provenance": _optional_string(record.get("provenance")),
        "authority": _source_authority(record.get("authority")),
        "freshness": _optional_string(record.get("freshness")),
        "lifecycle": _source_lifecycle(record.get("lifecycle")),
    }


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _source_kind(value: object) -> SourceKind:
    if value is None:
        return "document"
    if is_source_kind(value):
        return value
    raise ValueError(f"Invalid source_kind: {value!r}")


def _source_authority(value: object) -> SourceAuthority:
    if value is None:
        return "source"
    if is_source_authority(value):
        return value
    raise ValueError(f"Invalid source authority: {value!r}")


def _source_lifecycle(value: object) -> SourceLifecycle:
    if value is None:
        return "regenerable"
    if is_source_lifecycle(value):
        return value
    raise ValueError(f"Invalid source lifecycle: {value!r}")


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _publish_generation_directory(source: Path, destination: Path) -> None:
    source.replace(destination)
