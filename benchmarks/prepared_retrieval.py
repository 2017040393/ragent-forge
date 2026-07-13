from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from ragent_forge.app.services.evaluation.metrics import percentile
from ragent_forge.app.services.search_service import BM25SearchService
from ragent_forge.core.retrieval.contracts import ChunkRecord

DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "prepared_retrieval_manifest.json"
)


class BenchmarkWorkload(BaseModel):
    chunk_count: int = Field(gt=0)
    words_per_chunk: int = Field(gt=0)
    warm_runs: int = Field(gt=0)
    limit: int = Field(gt=0)
    queries: list[str] = Field(min_length=1)


class BenchmarkGates(BaseModel):
    max_workspace_reads: int = Field(ge=1)
    max_chunk_loads: int = Field(ge=1)
    minimum_warm_hits: int = Field(ge=1)


class BenchmarkManifest(BaseModel):
    schema_version: int = 1
    name: str
    description: str
    workload: BenchmarkWorkload
    gates: BenchmarkGates


class BenchmarkWorkspace:
    def __init__(self, records: list[ChunkRecord]) -> None:
        self.records = records
        self.read_chunks_calls = 0

    def read_chunks(self) -> list[ChunkRecord]:
        self.read_chunks_calls += 1
        return list(self.records)

    def current_snapshot_id(self) -> str:
        return "benchmark-snapshot-v1"


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> BenchmarkManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkManifest.model_validate(payload)


def run_benchmark(manifest: BenchmarkManifest) -> dict[str, object]:
    workspace = BenchmarkWorkspace(_build_chunks(manifest.workload))
    service = BM25SearchService(workspace)
    query = manifest.workload.queries[0]

    started = perf_counter()
    cold_results = service.search(query, manifest.workload.limit)
    cold_ms = (perf_counter() - started) * 1000

    warm_timings_ms: list[float] = []
    warm_result_counts: list[int] = []
    for run_index in range(manifest.workload.warm_runs):
        query = manifest.workload.queries[
            run_index % len(manifest.workload.queries)
        ]
        started = perf_counter()
        results = service.search(query, manifest.workload.limit)
        warm_timings_ms.append((perf_counter() - started) * 1000)
        warm_result_counts.append(len(results))

    stats = service.prepared_state_cache.stats()
    gates = {
        "workspace_reads": (
            workspace.read_chunks_calls <= manifest.gates.max_workspace_reads
        ),
        "chunk_loads": stats.chunk_loads <= manifest.gates.max_chunk_loads,
        "warm_hits": stats.warm_hits >= manifest.gates.minimum_warm_hits,
    }
    warm_p50_ms = statistics.median(warm_timings_ms)
    return {
        "schema_version": manifest.schema_version,
        "benchmark": manifest.name,
        "description": manifest.description,
        "measured_at": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "workload": manifest.workload.model_dump(mode="json"),
        "timings_ms": {
            "cold": round(cold_ms, 4),
            "warm_p50": round(warm_p50_ms, 4),
            "warm_p95": round(percentile(warm_timings_ms, 0.95), 4),
            "cold_to_warm_p50_ratio": (
                round(cold_ms / warm_p50_ms, 4)
                if warm_p50_ms > 0
                else None
            ),
        },
        "result_counts": {
            "cold": len(cold_results),
            "warm_min": min(warm_result_counts),
            "warm_max": max(warm_result_counts),
        },
        "cache": {
            "workspace_reads": workspace.read_chunks_calls,
            "chunk_loads": stats.chunk_loads,
            "warm_hits": stats.warm_hits,
            "invalidations": stats.invalidations,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _build_chunks(workload: BenchmarkWorkload) -> list[ChunkRecord]:
    topics = (
        "agent memory retrieval",
        "hybrid context selection",
        "workspace snapshot trace",
        "document evidence provenance",
    )
    records: list[ChunkRecord] = []
    for index in range(workload.chunk_count):
        topic = topics[index % len(topics)]
        words = (f"document {index} {topic} local inspectable rag ".split())
        repeated = (words * ((workload.words_per_chunk // len(words)) + 1))[
            : workload.words_per_chunk
        ]
        records.append(
            {
                "schema_version": 2,
                "snapshot_id": "benchmark-snapshot-v1",
                "chunk_id": f"benchmark::chunk-{index:06d}",
                "document_id": f"benchmark-document-{index:06d}",
                "text": " ".join(repeated),
                "source_path": f"benchmark/document-{index:06d}.md",
                "start_char": 0,
                "end_char": len(" ".join(repeated)),
                "metadata": {},
                "source_kind": "document",
                "provenance": "generated-benchmark",
                "authority": "source",
                "freshness": None,
                "lifecycle": "regenerable",
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the prepared retrieval cold/warm benchmark."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the benchmark manifest JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the machine-readable JSON result.",
    )
    args = parser.parse_args(argv)
    result = run_benchmark(load_manifest(args.manifest))
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0 if result["passed"] is True else 1


if __name__ == "__main__":
    sys.exit(main())
