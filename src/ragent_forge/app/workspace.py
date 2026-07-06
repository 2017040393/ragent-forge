from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragent_forge.app.models import (
    DocumentChunk,
    IngestResult,
    OperationTrace,
    WorkspaceStatus,
)


class LocalWorkspace:
    def __init__(self, root_path: str | Path = ".ragent") -> None:
        self.root_path = Path(root_path).expanduser()
        self.chunks_dir = self.root_path / "chunks"
        self.ingest_dir = self.root_path / "ingest"
        self.traces_dir = self.root_path / "traces"
        self.index_dir = self.root_path / "index"
        self.eval_dir = self.root_path / "eval"
        self.eval_runs_dir = self.eval_dir / "runs"
        self.chunks_path = self.chunks_dir / "chunks.jsonl"
        self.latest_summary_path = self.ingest_dir / "latest_summary.json"
        self.latest_trace_path = self.traces_dir / "latest_trace.json"
        self.latest_retrieval_eval_path = (
            self.eval_dir / "latest_retrieval_eval.json"
        )
        self.config_path = self.root_path / "config.toml"
        self.vector_index_path = self.index_dir / "vector_index.jsonl"
        self.vector_index_manifest_path = (
            self.index_dir / "vector_index_manifest.json"
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

    def write_trace(self, trace: OperationTrace) -> Path:
        self.ensure_exists()
        content = (
            trace.model_dump_json(indent=2, exclude_none=True)
            + "\n"
        )
        trace_path = self.traces_dir / f"{trace.trace_id}.json"
        trace_path.write_text(content, encoding="utf-8")
        self.latest_trace_path.write_text(content, encoding="utf-8")
        return self.latest_trace_path

    def write_retrieval_eval_report(
        self,
        report: dict[str, Any],
        report_path: str | Path | None = None,
    ) -> Path:
        self.ensure_exists()
        if report_path is None:
            destination = self.eval_dir / (
                "retrieval_eval_"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
            )
        else:
            destination = Path(report_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        destination.write_text(content, encoding="utf-8")
        self.latest_retrieval_eval_path.write_text(content, encoding="utf-8")
        return destination

    def write_retrieval_eval_run(
        self,
        report: dict[str, Any],
        report_path: str | Path | None = None,
    ) -> Path:
        self.ensure_exists()
        run_dir = self._next_retrieval_eval_run_dir()
        run_dir.mkdir(parents=True)

        summary_json_path = run_dir / "summary.json"
        summary_md_path = run_dir / "summary.md"
        cases_jsonl_path = run_dir / "cases.jsonl"
        failures_jsonl_path = run_dir / "failures.jsonl"

        summary_json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_md_path.write_text(
            self._retrieval_eval_summary_markdown(
                report,
                summary_json_path=summary_json_path,
                cases_jsonl_path=cases_jsonl_path,
                failures_jsonl_path=failures_jsonl_path,
                report_path=Path(report_path) if report_path is not None else None,
            ),
            encoding="utf-8",
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
        return trace

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

    def _next_retrieval_eval_run_dir(self) -> Path:
        base_name = f"retrieval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        candidate = self.eval_runs_dir / base_name
        index = 1
        while candidate.exists():
            candidate = self.eval_runs_dir / f"{base_name}-{index:03d}"
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
            "recall",
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
        path.write_text(content, encoding="utf-8")


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
