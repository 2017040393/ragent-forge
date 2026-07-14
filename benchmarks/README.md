# Retrieval Benchmarks

## Formal Pre-v0.3 Baseline

`retrieval_baseline_manifest.json` freezes the v0.2 50-case dataset, corpus
hashes, ingest parameters, embedding configuration, retrieval modes, cutoffs,
and repetition count used for the post-architecture-convergence baseline.
Text-file hashes canonicalize CRLF and CR line endings to LF; binary source
hashes use the exact bytes. This keeps the manifest stable across Git checkout
settings without changing the historical v0.2 dataset manifest.

Prepare a dedicated generation-layout workspace from the repository root:

```powershell
$workspace = ".ragent/baselines/pre-v0.3"
uv run ragent ingest examples/knowledge --workspace $workspace
# Configure the same embedding provider in $workspace/config.toml.
uv run ragent index build --workspace $workspace
```

Run the baseline from a clean Git commit and choose a new output directory:

```powershell
uv run --extra dev python -m benchmarks.retrieval_baseline `
  --workspace $workspace `
  --output-dir benchmarks/results/pre-v0.3-<commit>
```

Each retrieval mode runs three isolated trials at top-20. The harness derives
quality and selected-context metrics at 5, 10, and 20 from the same ranking,
so changing a cutoff does not repeat external query-embedding calls. Within a
trial, the first query is the cold-start sample and the remaining queries are
the warm samples. Modes and trials never share prepared retrieval state.

The output contains a resolved `summary.json`, a copy of the manifest, and one
full machine-readable report per trial under `runs/`. It records the Git
commit, clean/dirty state, dataset and corpus hashes, workspace snapshot,
non-secret embedding settings, dependency versions, ranking fingerprints,
stage timings, and prepared-cache reuse. Diagnostic dirty-tree runs require
`--allow-dirty` and deliberately fail the formal `passed` result.

The historical v0.2 baseline remains frozen. Do not overwrite its versioned
dataset or report when producing this post-convergence baseline.

## Prepared Retrieval Smoke Benchmark

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
