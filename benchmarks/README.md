# Prepared Retrieval Benchmark

This deterministic harness measures the snapshot-keyed BM25 prepared-state
cache separately for its first cold query and repeated warm queries. It is an
architecture/performance smoke benchmark, not the frozen v0.3 retrieval-quality
baseline described in `docs/roadmap.md`.

Run it from the repository root:

```powershell
uv run --extra dev python -m benchmarks.prepared_retrieval
```

To persist the machine-readable result:

```powershell
uv run --extra dev python -m benchmarks.prepared_retrieval `
  --output .ragent/eval/prepared-retrieval-benchmark.json
```

The checked-in manifest freezes corpus size, query set, retrieval limit, warm
run count, and structural cache gates. Timing values remain machine-specific;
the gates verify that warm runs reuse one workspace read and one prepared chunk
load without claiming an ANN or retrieval-quality improvement.
