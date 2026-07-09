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
- `mrr`: mean reciprocal rank of the first matching result.
- `avg_retrieval_latency_ms`: average time spent inside retrieval search.
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
mode      k   status   hit@k   recall@k   mrr     avg_latency_ms   failures
lexical   5   success  0.5000  0.4200     0.3900  3.2000           4
bm25      5   success  0.6500  0.5700     0.5100  4.1000           3
semantic  5   success  0.7000  0.6200     0.5600  18.3000          2
hybrid    5   success  0.7800  0.6900     0.6300  22.5000          1
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
