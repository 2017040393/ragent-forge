# RAGentForge Project Walkthrough

> Language: English | [中文](PROJECT_WALKTHROUGH.zh-CN.md)

## What This Demo Shows

This walkthrough demonstrates the current local RAG loop:

```text
ingest local documents
-> inspect chunks
-> run lexical retrieval
-> optionally build a semantic index
-> run semantic or hybrid retrieval
-> ask with sources
-> inspect traces
-> run retrieval evaluation
-> inspect results in the command-first TUI
```

The demo uses the checked-in files under `examples/knowledge` and
`examples/eval`.

## Prerequisites

- Python 3.11 or newer.
- `uv` installed.
- Run commands from the repository root.
- Optional: an embedding provider configured in `.ragent/config.toml` for
  semantic and hybrid retrieval.
- Optional: a generation provider configured in `.ragent/config.toml` for
  generated answers.

Install dependencies:

```bash
uv sync --extra dev
```

## Step 1: Prepare the Workspace

Start with a local workspace directory:

```bash
uv run ragent status --workspace .ragent
uv run ragent config show --workspace .ragent
```

If you want to create a config file before editing provider settings:

```bash
uv run ragent config init --workspace .ragent
```

The default config uses `generation.provider = "null"` and
`embedding.provider = "none"`.

## Step 2: Ingest Local Knowledge

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
```

This loads Markdown/TXT files, creates deterministic chunks, writes
`.ragent/chunks/chunks.jsonl`, writes an ingestion summary, and writes a CLI
operation trace.

## Step 3: Inspect Workspace Status

```bash
uv run ragent status --workspace .ragent
```

Look for `Status: ready`, the document count, the chunk count, and paths to the
chunk and summary files.

## Step 4: Inspect Chunks

```bash
uv run ragent chunks list --workspace .ragent
uv run ragent chunks show "<chunk_id>" --workspace .ragent
```

Copy a chunk id from `chunks list` into `chunks show` when running the demo.
This proves the chunking output is inspectable.

## Step 5: Run Lexical Search

```bash
uv run ragent search "What is RAG?" --retrieval lexical --workspace .ragent
```

Lexical retrieval is the default mode and does not require embeddings or a
vector index.

## Step 6: Optional Semantic Index Build

Semantic and hybrid retrieval require an embedding provider and a vector index.
Configure `[embedding]` in `.ragent/config.toml`, then run:

```bash
uv run ragent index build --workspace .ragent
uv run ragent index status --workspace .ragent
```

The index is stored locally under `.ragent/index/` as JSONL plus a manifest.

## Step 7: Run Semantic and Hybrid Search

After the index exists:

```bash
uv run ragent search "What is Agentic RAG?" --retrieval semantic --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

Semantic search uses local vector similarity. Hybrid search combines BM25
and semantic candidates with Reciprocal Rank Fusion.

## Step 8: Ask a Question

Lexical Ask works without an index:

```bash
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

Hybrid Ask requires the vector index:

```bash
uv run ragent ask "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

With the default `null` generation provider, Ask stays in retrieval-only mode.
When `generation.provider = "openai_responses"` is configured, CLI Ask can
generate a source-grounded answer.

To inspect the prompt without hiding the retrieval context:

```bash
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --show-prompt --workspace .ragent
```

CLI `ragent ask` writes an Ask trace. Shell Ask in the TUI does not write new
traces in v0.1.

## Step 9: Inspect Sources in the TUI

Launch the TUI from the repository root after preparing `.ragent`:

```bash
uv run ragent tui
```

The current TUI command does not take a `--workspace` argument; it reads the
default `.ragent` workspace in the current working directory.

Try this Shell sequence:

```text
/help
/search Agentic RAG
/source 2
/sources
/source next
/source prev
What is Agentic RAG?
/trace
/settings
/exit
```

`/sources` shows the current source list. `/source <rank>`, `/source next`,
and `/source prev` change the source shown in the Inspector.

The TUI intentionally avoids global single-key shortcuts such as `q`; use
`/exit`, `/quit`, or `/q` from the composer.

## Step 10: Inspect Traces

```bash
uv run ragent traces latest --workspace .ragent
uv run ragent traces list --workspace .ragent
uv run ragent traces show "<trace_id>" --workspace .ragent
```

Use `traces list` to find a trace id for `traces show`. CLI ingest, index
build, search, ask, and retrieval eval workflows write traces.

## Step 11: Generate and Run Retrieval Evaluation

You can evaluate retrieval with a checked-in JSONL file or generate a fresh
span-based dataset from the source documents.

Span-based generation is the more flexible path when you want to compare
retrieval or chunking strategies. The generated cases point at stable source
evidence spans instead of fixed chunk ids, and `eval retrieval` maps those
spans onto whatever chunks exist in the current workspace.

The loop is:

1. Extract evidence spans and generate synthetic eval cases.
2. Run retrieval eval against current chunks.
3. Compare Hit@k, Recall@k, MRR, latency, and context-size metrics across
   retrieval or chunking strategies.

### Use Checked-In Cases

BM25 eval over the small demo case file:

```bash
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval bm25 --workspace .ragent --limit 5
```

### Generate Cases From Source Documents

`eval generate` extracts evidence spans from source documents, asks the
configured generation provider for synthetic questions, and writes JSONL cases
that `eval retrieval` can load.

Start with a dry run. This counts spans and estimates case count without calling
the generation provider:

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output .ragent/eval/generated_cases.jsonl --questions-per-span 2 --max-cases 10 --dry-run
```

Before generating for real, make sure `.ragent/config.toml` has a real
generation provider. The default `null` provider is fine for dry runs, search,
and retrieval eval, but real dataset generation requires
`generation.provider = "openai_responses"` plus its model, base URL, and API key.

Generate the dataset:

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output .ragent/eval/generated_cases.jsonl --questions-per-span 2 --max-cases 10 --overwrite
```

Then evaluate retrieval against the generated cases:

```bash
uv run ragent eval retrieval --cases .ragent/eval/generated_cases.jsonl --retrieval bm25 --workspace .ragent --limit 5
```

The generated cases reference evidence spans from the source documents. Run
`ragent ingest` on the same source before `eval retrieval` so the current
workspace chunks can be matched back to those spans.

For text-based PDFs, opt in explicitly:

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output .ragent/eval/generated_pdf_cases.jsonl --questions-per-span 2 --max-cases 10 --include-pdf --overwrite
```

### Evaluate Semantic or Hybrid Retrieval

Semantic and hybrid eval require a vector index built from the same workspace
chunks:

```bash
uv run ragent index build --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval semantic --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval hybrid --workspace .ragent
```

### Compare Retrieval Modes

Compare lexical, BM25, semantic, and hybrid retrieval across multiple top-k
limits:

```bash
uv run ragent eval compare --cases examples/eval/retrieval_cases.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5 --workspace .ragent
```

Semantic and hybrid runs require the vector index. Lexical and BM25 do not.

Retrieval eval reports Hit@k, Recall@k, MRR, latency, context-size metrics,
failure types, and failed cases. It does not evaluate answer quality and does
not run LLM-as-judge.

For the full v0.2 retrieval quality workflow, see
[RETRIEVAL_EVALUATION.md](RETRIEVAL_EVALUATION.md).

## What to Look For

- The source documents stay under `examples/knowledge`.
- Derived artifacts stay under `.ragent`.
- Chunks are readable JSONL records.
- Search and Ask output include source paths and chunk ids.
- Semantic and hybrid modes fail clearly until a vector index exists.
- CLI Ask writes traces; Shell Ask does not.
- `/trace` in the TUI reads the latest existing CLI trace.
- Retrieval eval uses small JSONL cases under `examples/eval`.

## Troubleshooting

- `No chunks found`: run `uv run ragent ingest examples/knowledge --workspace .ragent`.
- `vector index not found`: configure embeddings, then run
  `uv run ragent index build --workspace .ragent`.
- `generation.provider = null`: generated answers are disabled; Ask remains in
  retrieval-only mode.
- TUI shows no trace: run a CLI workflow such as `ingest`, `search`, `ask`, or
  `eval retrieval` first.
- Eval misses expected paths: inspect `uv run ragent chunks list --workspace .ragent`
  and make sure the case file uses either exact source paths or repo-relative
  suffixes such as `examples/knowledge/rag_basics.md`.

## Cleanup

The `.ragent/` directory contains derived local state. To restart the demo from
scratch, remove `.ragent/` and run the ingest command again.
