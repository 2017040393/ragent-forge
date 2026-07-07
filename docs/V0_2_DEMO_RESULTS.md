# RAGentForge v0.2 Demo Results Template

This file is a template for recording real local v0.2 demo output. Do not
invent metric values. Run the commands below in your local checkout and replace
the placeholders with the actual output from that run.

## Environment

Fill this section from your local run:

| Field | Value |
|---|---|
| Date | `<YYYY-MM-DD>` |
| Git commit | `<git rev-parse --short HEAD>` |
| Python | `<python version>` |
| OS | `<operating system>` |
| Workspace | `.ragent` |
| Corpus | `examples/knowledge` |
| Generation provider | `<provider or null>` |
| Embedding provider | `<provider or not configured>` |

## 1. Prepare Workspace

Run:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
uv run ragent chunks list --workspace .ragent --limit 10
```

Record the important output:

```text
<paste ingest/status/chunks summary here>
```

Notes:

- Confirm Markdown/TXT/PDF files were discovered as expected.
- Confirm chunks have stable source paths and useful metadata.
- If PDFs are present, record whether page/table metadata appears in chunk
  summaries or detailed chunk inspection.

## 2. Sparse Retrieval Baselines

Run:

```bash
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval bm25 --workspace .ragent
```

Record top results:

| Mode | Rank | Source | Chunk ID | Score | Notes |
|---|---:|---|---|---:|---|
| lexical | 1 | `<source>` | `<chunk-id>` | `<score>` | `<notes>` |
| lexical | 2 | `<source>` | `<chunk-id>` | `<score>` | `<notes>` |
| bm25 | 1 | `<source>` | `<chunk-id>` | `<score>` | `<notes>` |
| bm25 | 2 | `<source>` | `<chunk-id>` | `<score>` | `<notes>` |

## 3. Semantic and Hybrid Retrieval

Semantic and hybrid retrieval require an embedding provider and a vector index.
If embeddings are not configured, mark this section as skipped and record the
reason instead of inventing results.

Run:

```bash
uv run ragent index build --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval semantic --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

Record top results:

| Mode | Rank | Source | Chunk ID | Score | Notes |
|---|---:|---|---|---:|---|
| semantic | 1 | `<source>` | `<chunk-id>` | `<score>` | `<notes>` |
| semantic | 2 | `<source>` | `<chunk-id>` | `<score>` | `<notes>` |
| hybrid | 1 | `<source>` | `<chunk-id>` | `<score>` | `<notes>` |
| hybrid | 2 | `<source>` | `<chunk-id>` | `<score>` | `<notes>` |

## 4. Span-Grounded Eval Generation

First inspect evidence spans without calling a generation provider:

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --dry-run
```

If a generation provider is configured, generate real cases:

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --overwrite
```

If text-based PDFs should be included in evidence span extraction, run:

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --include-pdf --overwrite
```

Record generation output:

```text
<paste eval generate summary here>
```

Generated cases:

| Field | Value |
|---|---|
| Output file | `examples/eval/synthetic_span_cases.jsonl` |
| Generated cases | `<count>` |
| Skipped spans | `<count>` |
| Error count | `<count>` |
| Includes PDF evidence | `<yes/no>` |

## 5. Retrieval Eval

Run:

```bash
uv run ragent eval retrieval --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval bm25 --limit 5
```

Record the report paths:

| Artifact | Path |
|---|---|
| Latest compatibility report | `.ragent/eval/latest_retrieval_eval.json` |
| Run directory | `.ragent/eval/runs/<retrieval-run-id>` |
| Summary JSON | `.ragent/eval/runs/<retrieval-run-id>/summary.json` |
| Summary Markdown | `.ragent/eval/runs/<retrieval-run-id>/summary.md` |
| Cases JSONL | `.ragent/eval/runs/<retrieval-run-id>/cases.jsonl` |
| Failures JSONL | `.ragent/eval/runs/<retrieval-run-id>/failures.jsonl` |

Record metrics from the real run:

| Metric | Value |
|---|---:|
| cases | `<count>` |
| passed | `<count>` |
| failed | `<count>` |
| hit@1 | `<value>` |
| hit@3 | `<value>` |
| hit@5 | `<value>` |
| hit@k | `<value>` |
| recall@k | `<value>` |
| mrr | `<value>` |
| avg_retrieval_latency_ms | `<value>` |
| avg_retrieved_count | `<value>` |
| avg_retrieved_context_chars | `<value>` |
| avg_estimated_context_tokens | `<value>` |

Failure analysis:

| Failure Type | Count | Example Case ID | Notes |
|---|---:|---|---|
| `<failure_type>` | `<count>` | `<case-id>` | `<failure_reason>` |

## 6. Retrieval Compare

Run:

```bash
uv run ragent eval compare --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5
```

If semantic or hybrid cannot run because embeddings are not configured, record
the exact error and rerun a sparse-only compare:

```bash
uv run ragent eval compare --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval lexical,bm25 --limit 1,3,5
```

Record the real compare table:

| Retrieval | Limit | Cases | Hit@k | Recall@k | MRR | Avg Latency ms | Avg Context Chars | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| lexical | 1 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| lexical | 3 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| lexical | 5 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| bm25 | 1 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| bm25 | 3 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| bm25 | 5 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| semantic | 1 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| semantic | 3 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| semantic | 5 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| hybrid | 1 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| hybrid | 3 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |
| hybrid | 5 | `<count>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<notes>` |

## 7. TUI Smoke Check

Run:

```bash
uv run ragent tui
```

Inside the TUI, run:

```text
/search Agentic RAG
/mode bm25
/ask What does agentic RAG add?
/sources
/source next
/prompt on
/prompt off
/exit
```

Record observations:

| Check | Result | Notes |
|---|---|---|
| Search completes | `<pass/fail>` | `<notes>` |
| BM25 mode is selectable | `<pass/fail>` | `<notes>` |
| Sources are navigable | `<pass/fail>` | `<notes>` |
| Inspector shows selected source | `<pass/fail>` | `<notes>` |
| Prompt preview toggles | `<pass/fail>` | `<notes>` |

## 8. Final Notes

Use this section for demo interpretation:

- Best retrieval mode in this local run: `<mode>`
- Most useful failure type: `<failure_type>`
- Follow-up dataset cleanup needed: `<notes>`
- Follow-up retrieval improvement needed: `<notes>`
