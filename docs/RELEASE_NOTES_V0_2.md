# RAGentForge v0.2 Release Notes

## Retrieval Quality Foundation

RAGentForge v0.2 moves the project from a basic inspectable RAG demo toward a
retrieval quality engineering tool. It adds span-grounded eval generation,
retrieval metrics, reproducible run reports, deterministic failure analysis,
mode comparison, and a BM25 baseline.

## Added

- Span-based synthetic eval generation from source evidence.
- Evidence span to current chunk mapping during retrieval eval.
- Retrieval eval runner with Hit@k, Recall@k, MRR, latency, and context-size
  metrics.
- Persisted retrieval eval run reports:
  - `summary.json`
  - `summary.md`
  - `cases.jsonl`
  - `failures.jsonl`
- Failure analysis with `failure_type` and `failure_reason`.
- Retrieval compare runner for multiple retrieval modes and top-k limits.
- BM25 retrieval baseline.
- Local JSON/JSONL eval artifacts for review and automation.

## Changed

- Retrieval eval cases can now use `evidence_spans` instead of only fixed chunk
  ids or source paths.
- `ragent eval retrieval` writes both the latest compatibility report and a
  timestamped run directory.
- `ragent eval compare` can evaluate multiple retrieval modes and top-k limits
  in one command.
- Retrieval modes now include:
  - `lexical`
  - `bm25`
  - `semantic`
  - `hybrid`

## Notes

- Semantic and hybrid retrieval require a vector index:

```bash
uv run ragent index build --workspace .ragent
```

- BM25 and lexical do not require embeddings or a vector index.
- Span-grounded eval cases are mapped to the current chunk index at evaluation
  time, so they remain useful when chunking settings change.
- `eval generate --dry-run` does not call a model.
- Real synthetic question generation requires a configured generation provider.

## Deferred

- Reranking.
- Query rewriting.
- Agentic multi-step retrieval.
- LLM-as-judge answer grading.
- RAGAS integration.
- Web dashboard.
