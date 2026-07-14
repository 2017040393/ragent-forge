# Retrieval Evaluation

> Language: English | [中文](RETRIEVAL_EVALUATION.zh-CN.md)

RAGentForge v0.2 turns retrieval evaluation into a local quality engineering
loop. The goal is to make retrieval quality measurable, comparable, and
diagnosable without a hosted service or hidden backend.

The key idea is span-grounded evaluation. Eval datasets should not be tightly
coupled to one specific chunking strategy. RAGentForge can generate and
evaluate span-grounded retrieval eval cases. Evidence spans are mapped to the
current chunk index at evaluation time, so eval cases remain stable even when
chunking strategy changes.

## Local Workflow

Run commands from the repository root.

### 1. Ingest Documents

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
```

### 2. Optional Vector Index

Semantic and hybrid retrieval require a vector index. BM25 and lexical do not.

```bash
uv run ragent config init --workspace .ragent
# edit .ragent/config.toml for an embedding provider if needed
uv run ragent index build --workspace .ragent
```

### 3. Generate Span-Based Eval Cases

Start with a dry run. It extracts and counts evidence spans without calling a
model.

```bash
uv run ragent eval generate \
  --source examples/knowledge \
  --workspace .ragent \
  --output examples/eval/synthetic_span_cases.jsonl \
  --dry-run
```

Real generation requires a configured text generation provider, such as
`generation.provider = "openai_responses"` in `.ragent/config.toml`.

```bash
uv run ragent eval generate \
  --source examples/knowledge \
  --workspace .ragent \
  --output examples/eval/synthetic_span_cases.jsonl \
  --questions-per-span 2 \
  --max-cases 20 \
  --overwrite
```

Add `--include-pdf` when you want generated cases from text-based PDFs.

### 4. Run Retrieval Eval

```bash
uv run ragent eval retrieval \
  --workspace .ragent \
  --cases examples/eval/synthetic_span_cases.jsonl \
  --retrieval bm25 \
  --limit 5
```

### 5. Compare Retrieval Modes

```bash
uv run ragent eval compare \
  --workspace .ragent \
  --cases examples/eval/synthetic_span_cases.jsonl \
  --retrieval lexical,bm25,semantic,hybrid \
  --limit 1,3,5
```

If a semantic or hybrid run is requested before the vector index exists, the
compare report records that run as failed and continues unless `--fail-fast` is
used.

### Formal Baseline Runs

Routine `eval compare` calls may share prepared state across requested modes
and limits. Use the checked-in baseline harness when results must support v0.3
quality or efficiency gates:

```powershell
uv run --extra dev python -m benchmarks.retrieval_baseline `
  --workspace .ragent/baselines/pre-v0.3 `
  --workspace-build-commit <workspace-commit> `
  --output-dir benchmarks/results/pre-v0.3-<commit>
```

The harness validates the frozen dataset and corpus hashes, requires a
generation-layout workspace and matching vector index, isolates every trial,
and reports cold and warm latency separately for each requested mode and
limit. See `benchmarks/README.md` for the complete preparation, dense
variability policy, and artifact contract.

## Retrieval Modes

- `lexical`: simple token-overlap baseline.
- `bm25`: stronger lexical baseline using BM25 scoring. It does not require a
  vector index.
- `semantic`: embedding-based vector retrieval. It requires
  `uv run ragent index build --workspace .ragent`.
- `hybrid`: Reciprocal-rank-fusion style combination of BM25 and semantic
  retrieval. It requires the same vector index as semantic retrieval.

The Textual Shell TUI remains command-first and read-oriented for retrieval
workflows. It defaults to `hybrid`, and its `/mode` command supports
`lexical`, `bm25`, `semantic`, and `hybrid`.

## Evaluation Artifacts

Retrieval eval writes a compatibility report and a reproducible run directory:

```text
.ragent/eval/latest_retrieval_eval.json
.ragent/eval/runs/retrieval-YYYYMMDDTHHMMSSZ/
  summary.json
  summary.md
  cases.jsonl
  failures.jsonl
```

- `summary.json`: full machine-readable retrieval eval report.
- `summary.md`: human-readable run summary with metrics, report paths, and
  failure breakdown.
- `cases.jsonl`: compact evaluated cases without full retrieved chunk text.
- `failures.jsonl`: failed cases only, including `failure_type` and
  `failure_reason`.

Retrieval compare writes:

```text
.ragent/eval/latest_retrieval_compare.json
```

`latest_retrieval_compare.json` summarizes each requested retrieval mode and
top-k run, including metrics, status, run paths, and failures.

## Retrieval Metrics

- `hit@k`: whether a case has any matching expected chunk or expected source
  within the top-k retrieved results.
- `recall@k`: fraction of expected chunks retrieved within top-k. If a case has
  no expected chunks, recall is `0.0`.
- `precision@1/3/5/k`: mean fraction of the top-k slots occupied by relevant
  results. Chunk-grounded cases use unique expected chunk IDs. Source-only
  cases count the first match for each expected source, so repeated chunks from
  one source do not inflate precision.
- `ndcg@k`: binary normalized discounted cumulative gain. It rewards relevant
  results more when they appear near the top and supports multiple expected
  chunks.
- `mrr`: mean reciprocal rank of the first matching result.
- `evidence_coverage@k`: mean fraction of each evidence span covered by the
  retrieved results. Character offsets are used when available; PDF page
  overlap is the fallback. It is averaged only across cases with computable
  span geometry.
- `evidence_coverage_case_rate`: fraction of eval cases included in the
  evidence-coverage average.
- `mapping_coverage`: fraction of evidence spans successfully mapped to the
  current chunks, averaged across span-grounded cases.
- `mapping_coverage_case_rate`: fraction of eval cases that contain evidence
  spans and therefore participate in mapping coverage.
- `context_evidence_density`: fraction of retrieved context characters that
  come from relevant results.
- `duplicate_context_ratio`: repeated normalized 20-character text shingles
  across retrieved chunks divided by all per-chunk shingles. This detects
  duplicated or strongly overlapping context without treating all chunks on
  one PDF page as duplicates.
- `avg_retrieval_latency_ms`: average time spent inside retrieval search.
- `retrieval_latency_p50_ms` and `retrieval_latency_p95_ms`: median and
  long-tail retrieval latency, using linear percentile interpolation.
- `avg_retrieved_context_chars`: average retrieved context size in characters.
- `avg_estimated_context_tokens`: simple context cost estimate based on
  characters divided by 4.

These metrics measure retrieval behavior only. They do not grade answer
quality.

## Failure Analysis

Failure analysis is deterministic and intended for debugging. It is not
LLM-as-judge.

- `no_result`: no retrieval results returned.
- `unmapped_evidence`: evidence spans could not be mapped to current chunks.
- `missed_source`: retrieved results did not include any expected source path.
- `wrong_section`: expected source was retrieved, but no expected chunk was
  found in top-k.
- `low_rank`: expected chunks were not found within the evaluated top-k
  results.
- `unknown`: no deterministic failure heuristic matched.

Use `failures.jsonl` when you want to inspect individual failures. Use the
failure breakdown in `summary.md` or `latest_retrieval_compare.json` when you
want to spot recurring failure modes.

## Example Compare Table

This table is illustrative, not a checked-in benchmark result.

```text
mode      k   status   hit@k  rec@k  pre@k  nDCG   MRR    p95ms      fail
lexical   5   success  0.5000 0.4200 0.1800 0.4400 0.3900 4.8000     4
bm25      5   success  0.6500 0.5700 0.2600 0.5900 0.5100 6.2000     3
semantic  5   success  0.7000 0.6200 0.2800 0.6400 0.5600 24.1000    2
hybrid    5   success  0.7800 0.6900 0.3200 0.7100 0.6300 29.4000    1
```

## Demo Script

Use this narrative in an interview or project walkthrough:

1. Ingest a small local knowledge base.
2. Generate span-grounded eval cases from source documents.
3. Run BM25 retrieval eval without embeddings.
4. Compare lexical, BM25, semantic, and hybrid retrieval.
5. Open `failures.jsonl` and explain the failure types.
6. Explain why evidence spans keep eval cases independent from chunking.

## v0.2 Non-Goals

RAGentForge v0.2 intentionally does not include:

- Reranking.
- Query rewriting.
- Agentic multi-step retrieval.
- LLM-as-judge answer grading.
- RAGAS integration.
- Web dashboard.

These are future directions, not current capabilities.
