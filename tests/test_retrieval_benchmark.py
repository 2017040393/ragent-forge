from benchmarks.prepared_retrieval import (
    BenchmarkGates,
    BenchmarkManifest,
    BenchmarkWorkload,
    load_manifest,
    run_benchmark,
)


def test_checked_in_prepared_retrieval_manifest_is_valid() -> None:
    manifest = load_manifest()

    assert manifest.schema_version == 1
    assert manifest.workload.chunk_count == 5000
    assert manifest.gates.minimum_warm_hits == manifest.workload.warm_runs


def test_prepared_retrieval_benchmark_reuses_snapshot_cache() -> None:
    manifest = BenchmarkManifest(
        name="test-prepared-retrieval",
        description="Small deterministic test workload.",
        workload=BenchmarkWorkload(
            chunk_count=40,
            words_per_chunk=16,
            warm_runs=3,
            limit=5,
            queries=["agent memory", "workspace trace"],
        ),
        gates=BenchmarkGates(
            max_workspace_reads=1,
            max_chunk_loads=1,
            minimum_warm_hits=3,
        ),
    )

    result = run_benchmark(manifest)

    assert result["passed"] is True
    assert result["cache"] == {
        "workspace_reads": 1,
        "chunk_loads": 1,
        "warm_hits": 3,
        "invalidations": 0,
    }
