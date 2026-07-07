# RAGentForge v0.2 Release Notes

## Summary

RAGentForge v0.2 turns the project from a local inspectable RAG demo into a
retrieval quality engineering foundation. It adds BM25, span-grounded eval
generation, evidence-to-current-chunk mapping, richer retrieval metrics,
persisted eval reports, deterministic failure analysis, retrieval compare, and
a more polished command-first TUI inspection workflow.

The release keeps the system local-first and explicit: generated state remains
under `.ragent`, retrieval modes are selected by command, and eval artifacts are
plain JSON/JSONL/Markdown files that can be inspected, committed, or compared.

## Why It Matters

RAG quality work needs more than a single answer transcript. Teams need to know
which sources were retrieved, whether expected evidence was found, how much
context was assembled, how slow retrieval was, and whether misses are caused by
chunking, retrieval mode, ranking, or dataset design.

v0.2 makes that workflow repeatable. Eval cases can point at stable source
evidence instead of fixed chunk ids. At evaluation time, RAGentForge maps those
evidence spans to the current chunk store, so the same dataset remains useful
when chunk size, chunk overlap, ingestion, or retrieval strategy changes.

## Added

- Markdown/TXT/PDF structured ingestion through the shared
  `DocumentBlock[] -> BlockChunker -> DocumentChunk[]` pipeline.
- BM25 retrieval as a stronger sparse baseline that does not require embeddings.
- Hybrid retrieval that combines BM25 and semantic results with reciprocal rank
  fusion.
- Span-grounded synthetic eval generation from stable source evidence.
- Evidence-to-current-chunk mapping during retrieval eval.
- Retrieval eval metrics:
  - Hit@k
  - Recall@k
  - MRR
  - retrieval latency
  - retrieved context characters
  - estimated context tokens
- Persisted retrieval eval run directories under `.ragent/eval/runs/`:
  - `summary.json`
  - `summary.md`
  - `cases.jsonl`
  - `failures.jsonl`
- Deterministic failure analysis with `failure_type` and `failure_reason`.
- Retrieval compare for evaluating multiple retrieval modes and top-k limits in
  one command.
- Command-first TUI polish for source navigation, inspector context, visual
  theme, BM25 mode selection, and prompt preview.
- Local JSON/JSONL/Markdown eval artifacts suitable for review and automation.

## Changed

- `ragent eval retrieval` still writes the latest compatibility report, and now
  also writes a timestamped reproducible run directory.
- Retrieval eval cases can use `evidence_spans` in addition to older
  chunk/source expectations.
- Eval reports now include compact per-case records without full chunk text,
  embeddings, or provider secrets.
- `ragent eval compare` can evaluate multiple retrieval modes and top-k limits
  from one JSONL cases file.
- `hybrid` retrieval now means BM25 plus semantic retrieval. Semantic and
  hybrid modes require a vector index; lexical and BM25 do not.
- The TUI remains read-only for ingest, index, eval, and config mutation, while
  improving search/ask/source inspection ergonomics.

## How To Try It

Start with a clean local workspace:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
uv run ragent chunks list --workspace .ragent --limit 10
```

Run sparse retrieval baselines:

```bash
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval bm25 --workspace .ragent
```

Build a vector index before semantic or hybrid retrieval:

```bash
uv run ragent index build --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval semantic --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

Generate span-grounded eval cases. Use `--dry-run` first to inspect source
evidence without calling a generation provider:

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --dry-run
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --overwrite
```

Add `--include-pdf` when you want eval generation to include text-based PDFs:

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --include-pdf --overwrite
```

Evaluate retrieval and compare modes:

```bash
uv run ragent eval retrieval --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval bm25 --limit 5
uv run ragent eval compare --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5
```

Inspect the command-first TUI:

```bash
uv run ragent tui
```

Inside the TUI, try:

```text
/search Agentic RAG
/mode bm25
/ask What does agentic RAG add?
/sources
/source next
/prompt on
/exit
```

A measured local demo run with screenshots is recorded in
[V0_2_DEMO_RESULTS.md](V0_2_DEMO_RESULTS.md).

## Known Limitations

- Semantic and hybrid retrieval require a configured embedding provider and a
  built vector index.
- Synthetic question generation requires a configured generation provider unless
  `--dry-run` is used.
- PDF support targets text-based PDFs. OCR, scanned PDFs, image text
  recognition, and PDF rendering are not included.
- Markdown parsing is intentionally lightweight and line-based, not full
  CommonMark.
- Eval metrics measure retrieval behavior, not final answer quality.
- The TUI is intentionally read-only for ingest, index, eval, and config
  mutation workflows.

## Deferred Work

- Reranking and cross-encoder reranking.
- Query rewriting.
- Agentic multi-step retrieval.
- LLM-as-judge answer grading.
- RAGAS integration.
- OCR and scanned PDF support.
- PDF viewing/editing or source full-text viewing.
- Web dashboard.
- Short demo recordings and broader benchmark-style corpora.
