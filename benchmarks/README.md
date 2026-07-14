# Retrieval Benchmarks

## Formal Pre-v0.3 Baseline

The frozen post-architecture result is checked in at
[`results/pre-v0.3-20260714-c410e2e`](results/pre-v0.3-20260714-c410e2e).
See the
[`baseline report`](../ideas/2026-07-14-pre-v0-3-post-architecture-baseline.md)
for the resolved quality, cold/warm latency, dense-provider variability, and
absolute v0.3 release-gate values.

`retrieval_baseline_manifest.json` freezes the v0.2 50-case dataset, corpus
hashes, ingest parameters, embedding configuration, retrieval modes, limits,
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
  --workspace-build-commit <workspace-commit> `
  --output-dir benchmarks/results/pre-v0.3-<commit>
```

Each retrieval mode runs three isolated trials at requested limits 5, 10, and
20. Limits are separate configurations because candidate depth and fusion can
depend on the requested limit. Within a trial, the first query is the
cold-start sample and the remaining queries are the warm samples. Modes,
limits, and trials never share prepared retrieval state.

Lexical and BM25 rankings must have identical fingerprints across repetitions.
External dense embeddings can vary slightly, so semantic and hybrid preserve
each fingerprint and report per-metric average, minimum, maximum, and spread.
The checked-in manifest freezes the maximum accepted quality-metric spread;
ranking variability is never hidden by the aggregate.

If an external provider interrupts a long matrix, rerun the same command with
`--resume`. The harness ignores that output directory when checking for
untracked files, then validates the copied manifest and every existing trial's
benchmark name, workspace snapshot, workspace build commit, mode, limit,
repetition, case count, and cache state before reusing it. Unexpected or
mismatched artifacts stop the resume; completed trials are never overwritten.
The final summary lists every Git commit that produced a trial.

The output contains a resolved `summary.json`, a copy of the manifest, and one
full machine-readable report per trial under `runs/`. It records the Git
evaluation commit, workspace build commit, clean/dirty state, dataset and
corpus hashes, workspace snapshot,
non-secret embedding settings, dependency versions, ranking fingerprints,
stage timings, and prepared-cache reuse. Diagnostic dirty-tree runs require
`--allow-dirty` and deliberately fail the formal `passed` result.

The historical v0.2 baseline remains frozen. Do not overwrite its versioned
dataset or report when producing this post-convergence baseline.

## Representation Screening

`retrieval_screen_manifest.json` defines the fixed 16-case diagnostic screen
used to reject weak representation variants before a full 50-case run. The
screen keeps the complete 1744-chunk corpus and independently evaluates
requested limits 5 and 20; it reduces query count, not the distractor corpus.

The selected cases are grouped as stable controls, semantic opportunities,
wrong-section challenges, hard misses, and observation-only boundary canaries.
Promotion is based on case transitions and explicit gates rather than the
slice-wide average, because this diagnostic slice is intentionally not a
representative benchmark sample.

Run the E0 screen against the workspace used for the formal baseline:

```powershell
uv run --extra dev python -m benchmarks.retrieval_screen `
  --workspace .ragent/baselines/pre-v0.3-ca029f9 `
  --output-dir .ragent/eval/screens/E0-raw-text
```

The four semantic/hybrid configurations share one prepared-state cache and a
query embedding cache. A later variant using the same query representation can
reuse the frozen vectors without mutating the source cache:

```powershell
uv run --extra dev python -m benchmarks.retrieval_screen `
  --manifest <candidate-manifest> `
  --workspace <candidate-workspace> `
  --query-cache-source <E0-result>/query_embeddings.json `
  --output-dir <candidate-result>
```

Use `--resume` only with the same manifest, Git commit, workspace snapshot, and
output directory. The runner validates every reused artifact before continuing.

Screen latency is diagnostic only: cache reuse deliberately changes the timing
boundary. A promoted candidate must still pass the full 50-case confirmation
and formal three-trial matrix before its quality or latency can be used for a
release decision. `BaselineWorkloadSpec` continues to require at least three
repetitions; the one-run screening contract is separate.

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
