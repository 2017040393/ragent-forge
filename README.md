# RAGentForge

> Language: English | [中文](README.zh-CN.md)

RAGentForge is a local-first, command-first RAG workbench for building,
inspecting, and evaluating retrieval-augmented systems. It focuses on
retrieval quality, grounded context construction, traceable runs, reproducible
evaluation, and retrieval mode comparison over Markdown/TXT/PDF knowledge
bases.

## What It Is

RAGentForge is a small Python project for developers who want to understand,
debug, and demo retrieval augmented generation workflows without a hosted
service or hidden backend. It stores generated state under a local `.ragent/`
workspace and exposes that state through CLI commands plus a Textual Shell TUI.

It is not a full autonomous agent framework. The current v0.2 surface is a
local, inspectable foundation for ingestion, retrieval, optional generation,
traces, span-grounded retrieval evaluation, retrieval comparison, and
command-first TUI inspection.

## Why It Exists

Many RAG demos hide the important engineering details behind a hosted app,
framework abstraction, or vector database. RAGentForge keeps the workflow
plain and inspectable so a reader can see the data flow from local documents to
chunks, retrieval results, context packs, answers, sources, traces, and eval
reports.

## Features

- Markdown/TXT ingestion from local files or folders.
- PDF ingestion with page text, table extraction, page ranges, and source
  quality metadata.
- Unified structured ingestion through `DocumentBlock[] -> BlockChunker` for
  Markdown, TXT, and PDF.
- Deterministic chunking into JSONL records with format-aware metadata.
- Local workspace storage under `.ragent/`.
- Lexical and BM25 retrieval over generated chunks.
- OpenAI-compatible embedding configuration for semantic retrieval.
- Local JSONL vector index for semantic search.
- Hybrid retrieval with Reciprocal Rank Fusion over BM25 and semantic
  candidates.
- Ask pipeline with optional OpenAI Responses-compatible generation.
- Retrieval-only Ask mode when generation is not configured.
- Source-grounded answers and compact source displays.
- Local operation traces for CLI ingest, index build, search, ask, and
  retrieval eval workflows.
- Span-based synthetic eval generation from stable source evidence instead of
  fixed chunk ids.
- Retrieval evaluation with Hit@k, Recall@k, Precision@k, nDCG@k, evidence
  coverage, latency percentiles, and context-quality metrics.
- Persisted retrieval eval run reports with compact case and failure JSONL.
- Deterministic failure analysis with `failure_type` and `failure_reason`.
- Retrieval comparison across lexical, BM25, semantic, and hybrid modes.
- Command-first Textual TUI Shell with default hybrid Ask, streaming answer
  display, saved sessions, source navigation, and an Inspector panel.

## Quickstart

Install development dependencies:

```bash
uv sync --extra dev
```

Prepare the sample workspace:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
uv run ragent chunks list --workspace .ragent
```

Inspect a chunk and its structured metadata:

```bash
uv run ragent chunks show "<chunk_id>" --workspace .ragent
```

Run lexical or BM25 retrieval:

```bash
uv run ragent search "What is RAG?" --retrieval lexical --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval bm25 --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

Launch the command-first TUI from the project root:

```bash
uv run ragent tui
uv run ragent tui --workspace .ragent
```

Without `--workspace`, the TUI reads the default `.ragent` workspace in the
current working directory. Use `--workspace` to inspect a different local
workspace. TUI sessions and exports are saved inside that workspace's
`sessions/` directory, which is `.ragent/sessions/` for the default workspace.
Use the CLI for ingest, index build, eval, and config editing.

## End-to-End Demo

For the reproducible demo flow, see
[docs/PROJECT_WALKTHROUGH.md](docs/PROJECT_WALKTHROUGH.md).

For the structured ingestion milestone demo, see
[docs/STRUCTURED_INGESTION_DEMO.md](docs/STRUCTURED_INGESTION_DEMO.md).

The short version:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent chunks list --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval bm25 --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent traces latest --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval bm25 --workspace .ragent --limit 5
uv run ragent eval compare --cases examples/eval/retrieval_cases.jsonl --retrieval lexical,bm25 --limit 1,3,5 --workspace .ragent
uv run ragent tui --workspace .ragent
```

## v0.2 Retrieval Quality Foundation

Goal: make retrieval quality measurable, comparable, and diagnosable.

The core idea is that eval datasets should not be tightly coupled to one
specific chunking strategy. RAGentForge can generate and evaluate
span-grounded retrieval eval cases. Evidence spans are mapped to the current
chunk index at evaluation time, so eval cases remain stable even when chunking
strategy changes.

v0.2 adds:

- Span-based synthetic eval generation.
- Evidence-to-current-chunk mapping.
- Retrieval eval runner with ranking, evidence coverage, latency percentile,
  and context-quality metrics.
- Persisted eval run reports under `.ragent/eval/runs/`.
- Deterministic failure analysis.
- Lexical, BM25, semantic, and hybrid retrieval comparison.
- Local JSON/JSONL artifacts for review and automation.

For the full workflow, metrics, artifacts, failure types, and demo script, see
[docs/RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md).

## Retrieval Evaluation Workflow

RAGentForge can generate retrieval evaluation cases from stable source
evidence spans instead of hand-written fixed chunk ids. This makes the dataset
useful across chunk-size, chunk-overlap, retrieval-mode, and ranking changes:
the cases keep pointing at source evidence, while `eval retrieval` maps that
evidence onto the current workspace chunks.

The workflow is:

1. Extract evidence spans and generate synthetic eval cases.
2. Run retrieval eval against the current chunks.
3. Compare metrics across retrieval and chunking strategies.

Use this compact local workflow:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --dry-run
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --overwrite
uv run ragent eval retrieval --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval bm25 --limit 5
uv run ragent eval compare --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5
```

`eval generate --dry-run` does not call a model. Without `--dry-run`,
`eval generate` extracts evidence spans directly from Markdown/TXT source files
and calls the configured generation provider. Add `--include-pdf` when the
source includes text-based PDFs. `eval retrieval` then maps those evidence spans
back to the current workspace chunks, so run `ragent ingest` on the same source
documents before evaluating.

The compare command above includes semantic and hybrid runs. Build the vector
index first if you want those runs to succeed; lexical and BM25 work without
embeddings.

Semantic and hybrid retrieval require an embedding provider in
`.ragent/config.toml` and a built vector index:

```bash
uv run ragent config init --workspace .ragent
uv run ragent index build --workspace .ragent
uv run ragent index status --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval semantic --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

With the default config, generation uses the `null` provider. In that mode
`ragent ask` retrieves and displays context but does not call a model.

## Retrieval Modes

- `lexical`: simple token-overlap baseline.
- `bm25`: stronger lexical baseline using BM25 scoring. It does not require a
  vector index.
- `semantic`: embedding-based vector retrieval. It requires
  `uv run ragent index build --workspace .ragent`.
- `hybrid`: Reciprocal-rank-fusion style combination of BM25 and semantic
  retrieval. It also requires a vector index.

Illustrative compare output:

```text
mode      k   status   hit@k  rec@k  pre@k  nDCG   MRR    p95ms      fail
lexical   5   success  0.5000 0.4200 0.1800 0.4400 0.3900 4.8000     4
bm25      5   success  0.6500 0.5700 0.2600 0.5900 0.5100 6.2000     3
semantic  5   success  0.7000 0.6200 0.2800 0.6400 0.5600 24.1000    2
hybrid    5   success  0.7800 0.6900 0.3200 0.7100 0.6300 29.4000    1
```

The numbers above are illustrative, not checked-in benchmark results.

## Screenshots

TUI Shell search with compact source results:

![TUI Shell search](docs/assets/v0_1/tui-shell-search.jpg)

Selected-source Inspector after source navigation:

![TUI source inspector](docs/assets/v0_1/tui-source-inspector.jpg)

Trace and settings inspection in the TUI:

![TUI trace and settings](docs/assets/v0_1/tui-trace-settings.jpg)

Retrieval evaluation output:

![Retrieval evaluation output](docs/assets/v0_1/tui-retrieval-eval.jpg)

## Command-First TUI

`uv run ragent tui --workspace .ragent` opens a single Shell interface with a
focused user/assistant transcript, composer, status line, inline command
suggestions, source/session pickers, and an Inspector for the selected answer
or source.

Ordinary text is Ask by default, so typing a question is equivalent to
`/ask <question>`. The default retrieval mode is `hybrid`, and generated answer
text streams into the transcript when the configured provider supports
streaming. `/search <query>` runs Shell Search in a background worker. The
Shell reads existing workspace chunks and indexes; it does not run ingest,
build the semantic index, run retrieval eval, or edit config.

The TUI restores the latest saved session on launch. Successful and failed Ask
runs are saved as assistant turns with their sources and run metadata under
`.ragent/sessions/`. The Inspector follows the selected answer, so `/turn`,
`/source`, and the source picker can review evidence for a specific response.
The main transcript stays focused on user questions and assistant replies;
retrieval details, selected-source previews, prompt preview state, and run
metadata live in the Inspector or command-result modals. Assistant replies use
small status badges such as `[1 source]` or `[failed]` instead of expanding
operational details into the chat surface.

After `/search <query>` returns results, the TUI opens the source picker so you
can choose evidence immediately. Source picker rows include location,
retrieval method, score, and chunk id; PDF rows preserve page-aware labels where
available. `/docs`, `/trace`, and `/settings` open read-only result modals and
also update the Inspector.

Useful Shell commands:

```text
/help
/mode lexical|bm25|semantic|hybrid
/limit <n>
/context <n>
/prompt on|off
/search <query>
/ask <question>
/sources
/source <rank>
/source next
/source prev
/sessions [recent|pinned|starred|failed|has-sources]
/new
/switch <session-id>
/rename <title>
/delete
/pin
/star
/session-search <query>
/export markdown|json
/branch
/rerun
/continue-sources
/title [auto|text]
/turn <id|number|next|prev|first|last>
/docs
/trace
/settings
/config
/clear
/exit
/quit
/q
```

Shell source navigation:

```text
/sources
/source <rank>
/source next
/source prev
```

Typing `/` opens inline command candidates. Use Up/Down to choose a command,
then Tab or Enter to complete it into the composer. Command execution still
happens through composer text. Argument suggestions show the current value when
available, for example the active retrieval mode in `/mode ` suggestions.

If an Ask or Search worker is running, the input remains editable. Submitting a
non-empty draft records `1 draft queued`; when the current request completes,
the draft stays in the composer with `1 draft ready` so you can press Enter to
send it. Worker failures print next-step hints such as `/settings`, `/docs`, or
`/mode bm25` instead of stack traces.

The TUI intentionally avoids global single-key shortcuts such as `q` to quit.
Use `/exit`, `/quit`, or `/q` from the composer.

Shell Ask writes TUI session artifacts, not operation traces. CLI
`uv run ragent ask ...` remains the trace-producing Ask workflow. Shell
`/trace` reads the latest trace already written by CLI workflows.

## Architecture

The project is organized around presentation layers, application services,
focused core modules, and local workspace storage. The high-level pipeline is:

```text
local documents
-> ingest
-> structured loaders
-> Document + DocumentBlock[]
-> BlockChunker
-> deterministic chunks
-> lexical / BM25 / semantic / hybrid retrieval
-> context pack
-> optional generation
-> answer + sources
-> traces
-> retrieval eval
-> command-first TUI inspection
-> saved TUI sessions and exports
```

Read the full architecture note:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The structured ingestion design note is here:
[docs/STRUCTURED_INGESTION_DESIGN.md](docs/STRUCTURED_INGESTION_DESIGN.md).

The TUI design note is here:
[docs/TUI_COMMAND_SHELL_DESIGN.md](docs/TUI_COMMAND_SHELL_DESIGN.md).

## Project Scope

The current scope is documented in
[docs/V0_1_SCOPE.md](docs/V0_1_SCOPE.md).

v0.1 includes local ingestion, deterministic chunks, lexical retrieval,
semantic retrieval, hybrid RRF retrieval, optional generation, source display,
traces, retrieval evaluation, and a command-first TUI Shell.

v0.2 adds the retrieval quality foundation: span-grounded eval generation,
evidence-to-chunk mapping, richer retrieval metrics, persisted eval run
reports, failure analysis, retrieval comparison, and BM25.

The current TUI also includes a local session workbench with saved conversations,
recent/pinned/starred/failed/has-sources filters, pin/star/search, session
export, branch/rerun helpers, selected-answer source inspection, source picker
inspection, actionable worker errors, queued drafts, and streaming Ask output.

## Release and Portfolio Materials

- [v0.1 Demo Script](docs/DEMO_SCRIPT.md)
- [v0.2 Retrieval Evaluation Guide](docs/RETRIEVAL_EVALUATION.md)
- [v0.2 Release Notes](docs/RELEASE_NOTES_V0_2.md)
- [v0.2 Demo Results](docs/V0_2_DEMO_RESULTS.md)
- [v0.1 Release Notes](docs/RELEASE_NOTES_V0_1.md)
- [v0.1-alpha-1 Structured Ingestion Release Notes](docs/RELEASE_NOTES_V0_1_ALPHA_1.md)
- [Structured Ingestion Demo Workflow](docs/STRUCTURED_INGESTION_DEMO.md)
- [Structured Ingestion Design](docs/STRUCTURED_INGESTION_DESIGN.md)
- [Portfolio Summary](docs/PORTFOLIO_SUMMARY.md)

## Current Limitations

RAGentForge v0.2 intentionally does not include reranking, cross-encoder
reranking, query rewriting, agentic multi-step retrieval, LLM-as-judge answer
grading, RAGAS integration, OCR/scanned PDF support, PDF viewing/editing, web
dashboard, vector databases, agent tool loops, or TUI write operations such as
ingest/index/eval/config editing.

Semantic and hybrid retrieval require a vector index. Generation depends on a
configured OpenAI Responses-compatible provider; otherwise Ask stays in
retrieval-only mode.

## Roadmap

v0.3 is the retrieval quality and efficiency engineering stage. It defines a
frozen benchmark and baseline-relative release gates before selecting technical
approaches, then targets measurable improvements in candidate recall, final
precision, retrieval latency, and context cost. Workspace-local project memory
joins document evidence, user notes, and session memory behind one retrieval
entry point. Each item keeps its source type, provenance, and lifecycle while
using the same stage-aware traces and evaluation. v0.4 builds
controlled multi-step retrieval and agent workflows on that pipeline. v0.5 adds
local comparison views for retrieval and answer quality. v0.3 is the active
development stage; v0.4 and v0.5 remain future directions.

More context:

- [Project ideas and learning notes](ideas/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [docs/roadmap.md](docs/roadmap.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md)
- [docs/PROJECT_WALKTHROUGH.md](docs/PROJECT_WALKTHROUGH.md)
- [docs/V0_1_SCOPE.md](docs/V0_1_SCOPE.md)
- [docs/TUI_COMMAND_SHELL_DESIGN.md](docs/TUI_COMMAND_SHELL_DESIGN.md)
- [docs/STRUCTURED_INGESTION_DESIGN.md](docs/STRUCTURED_INGESTION_DESIGN.md)
- [docs/STRUCTURED_INGESTION_DEMO.md](docs/STRUCTURED_INGESTION_DEMO.md)
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)
- [docs/RELEASE_NOTES_V0_2.md](docs/RELEASE_NOTES_V0_2.md)
- [docs/RELEASE_NOTES_V0_1.md](docs/RELEASE_NOTES_V0_1.md)
- [docs/RELEASE_NOTES_V0_1_ALPHA_1.md](docs/RELEASE_NOTES_V0_1_ALPHA_1.md)
- [docs/PORTFOLIO_SUMMARY.md](docs/PORTFOLIO_SUMMARY.md)
