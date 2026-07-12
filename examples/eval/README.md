# Eval Examples

This directory contains lightweight retrieval eval examples.

- `retrieval_cases.jsonl` is a small hand-written source-path eval file for
  quick local demos.
- `synthetic_span_cases.example.jsonl` shows the shape of span-grounded cases
  produced by `ragent eval generate`.
- `v0_2_retrieval_baseline.generated.jsonl` is the versioned, span-grounded
  50-case dataset used by the v0.2 retrieval baseline for v0.3 research. It
  preserves the original 20 cases and adds 30 cases sampled across both PDFs.
- `v0_2_retrieval_baseline.generated.manifest.json` records its generation
  model, source hashes, balanced span selection, and dataset checksum.

The versioned baseline dataset uses repository-relative source paths so its
evidence spans can be mapped after cloning the repository on another machine.
Do not overwrite it during routine local experiments; generate candidate data
under `.ragent/eval` instead.

Generate a fresh local dataset:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent eval generate \
  --source examples/knowledge \
  --workspace .ragent \
  --output .ragent/eval/generated_cases.jsonl \
  --questions-per-span 2 \
  --max-cases 20 \
  --overwrite
```

Start with `--dry-run` when you only want to count evidence spans and estimate
case volume without calling a model.

Evaluate or compare retrieval:

```bash
uv run ragent eval retrieval \
  --workspace .ragent \
  --cases .ragent/eval/generated_cases.jsonl \
  --retrieval bm25 \
  --limit 5

uv run ragent eval compare \
  --workspace .ragent \
  --cases .ragent/eval/generated_cases.jsonl \
  --retrieval lexical,bm25,semantic,hybrid \
  --limit 1,3,5
```

Semantic and hybrid comparison runs require `uv run ragent index build
--workspace .ragent`. Lexical and BM25 do not require embeddings.
