# RAGentForge Architecture

> Language: English | [中文](ARCHITECTURE.zh-CN.md)

## Overview

RAGentForge is a local-first, inspectable RAG console. It keeps the current
MVP small: local Markdown/TXT/PDF documents are ingested into deterministic
chunks, retrieved with lexical, BM25, semantic, or hybrid retrieval, assembled
into context, optionally sent to an OpenAI Responses-compatible generation
provider, and made inspectable through sources, traces, retrieval eval reports,
CLI commands, and a command-first TUI Shell.

## Design Goals

- Keep source documents local and user-owned.
- Store derived state in plain local files under `.ragent/`.
- Make every major RAG step inspectable.
- Keep CLI and TUI behavior backed by shared services.
- Prefer explicit commands over hidden automation.
- Avoid framework lock-in and heavy runtime dependencies in v0.2.

## High-Level Pipeline

```text
local documents
-> ingest
-> deterministic chunks
-> lexical / BM25 / semantic / hybrid retrieval
-> context pack
-> optional generation
-> answer + sources
-> traces
-> retrieval eval
-> command-first TUI inspection
```

## Layers

### CLI Layer

The CLI in `src/ragent_forge/cli.py` is the primary write-capable operator
surface. It runs ingestion, status inspection, chunk inspection, config
inspection/init, semantic index build/status, search, ask, trace inspection,
and retrieval evaluation.

CLI workflows write local artifacts such as chunks, summaries, vector indexes,
traces, and eval reports. CLI `ragent ask` is the trace-producing Ask workflow.

### TUI Shell Layer

The Textual Shell in `src/ragent_forge/tui/` is an inspectable command console,
not a management dashboard. It provides a composer, transcript, status line,
command suggestions, source/session pickers, selected-answer and
selected-source Inspector views, read-only summaries, Shell Search, and
streaming Shell Ask. The main transcript is intentionally chat-focused: it
renders user questions and assistant replies with small status badges such as
`[1 source]` or `[failed]`, while retrieval details stay in source pickers,
command-result modals, and the Inspector.

The Shell intentionally does not run ingest, build indexes, run eval, edit
config, or open local files. It does write local TUI session artifacts under
`.ragent/sessions/`, including saved turns, sources, run metadata, exports, and
the latest-session pointer. Shell Ask does not write operation traces; `/trace`
reads traces produced by CLI workflows. Running workers leave the composer
editable and can queue one draft; worker failures surface actionable next steps
such as `/settings`, `/docs`, or a sparse fallback like `/mode bm25`.

### Application Services

Application services under `src/ragent_forge/app/services/` coordinate use
cases such as ingesting files, listing chunks, loading config, building the
index, running retrieval, asking questions, writing traces, and evaluating
retrieval cases.

These services keep CLI and TUI code thin and reduce duplication between
presentation surfaces.

### Retrieval Services

Retrieval is explicit and mode-based:

- `lexical` uses deterministic token overlap over local chunks.
- `semantic` uses cosine similarity over the local JSONL vector index.
- `hybrid` combines BM25 and semantic candidates with Reciprocal Rank
  Fusion.

Semantic and hybrid retrieval require a vector index created by
`ragent index build`.

### Generation Services

Generation is optional. With the default `null` provider, Ask stays in
retrieval-only mode and prints retrieved context. When configured with
`openai_responses`, Ask sends a source-grounded prompt to
`{base_url}/responses`.

Chat Completions, answer evaluation, and LLM-as-judge are not part of the
current implementation. The CLI Ask path is still trace-oriented; the TUI Ask
path can stream provider deltas into its local transcript/session state.

### Workspace Storage

`LocalWorkspace` centralizes `.ragent/` paths and reads/writes derived state.
The source documents remain the source of truth; workspace files can be
regenerated.

Important workspace files include:

```text
.ragent/chunks/chunks.jsonl
.ragent/ingest/latest_summary.json
.ragent/config.toml
.ragent/index/vector_index.jsonl
.ragent/index/vector_index_manifest.json
.ragent/traces/latest_trace.json
.ragent/traces/<trace_id>.json
.ragent/eval/latest_retrieval_eval.json
.ragent/eval/retrieval_eval_<timestamp>.json
.ragent/eval/runs/<run-id>/
.ragent/sessions/latest.json
.ragent/sessions/index.json
.ragent/sessions/session-<id>.json
.ragent/sessions/exports/
```

### Trace and Evaluation

Traces are local JSON files that record compact metadata and workflow steps for
CLI operations. Retrieval evaluation reads JSONL cases and reports hit@1,
hit@3, hit@5, requested hit@k, MRR, and failed cases.

Retrieval eval is retrieval-only. It does not judge generated answer quality.

## Data Flow

### Ingestion Flow

`ragent ingest <path>` loads Markdown/TXT files, skips unsupported files,
chunks documents deterministically, writes chunk JSONL, writes the latest
ingestion summary, and writes an ingest trace.

### Search Flow

`ragent search <query>` reads chunks from the workspace, runs the selected
retrieval mode, prints ranked results with source paths and previews, and
writes a search trace.

Semantic and hybrid search fail clearly when the vector index is missing.

### Ask Flow

`ragent ask <question>` retrieves context, builds a context pack, optionally
builds a generation prompt, optionally calls the configured generation
provider, prints an answer or retrieved context with sources, and writes an
Ask retrieval trace.

If generation is not configured, the default `null` provider keeps Ask in
retrieval-only mode.

### Trace Flow

Trace commands read local trace files:

```text
ragent traces latest
ragent traces list
ragent traces show <trace_id>
```

The TUI `/trace` command shows a compact read-only summary of the latest trace.

### Evaluation Flow

`ragent eval generate --source <path>` loads supported source documents through
the structured ingestion loader, extracts stable evidence spans, calls the
configured text generation provider, and writes JSONL cases. Markdown and TXT
are included by default; text-based PDF extraction is opt-in with
`--include-pdf`.

Those generated cases are span-based rather than chunk-id-based. The eval
dataset can therefore survive chunk-size and chunk-overlap changes while still
checking whether the current retrieval system returns chunks that cover the
same source evidence.

`ragent eval retrieval --cases <path>` loads JSONL cases, runs the selected
retrieval mode, maps span-based cases back to current workspace chunks, checks
expected chunk ids or source paths, writes a compact report, and writes a
retrieval eval trace.

Semantic and hybrid eval require the same vector index as semantic and hybrid
search.

## Workspace Layout

The repository includes small demo inputs:

```text
examples/knowledge/
examples/eval/retrieval_cases.jsonl
```

The generated workspace defaults to:

```text
.ragent/
```

The TUI currently reads the default `.ragent` workspace from the current
working directory. Use CLI commands to prepare that workspace before launching
the TUI.

## Why Local-First

Local-first operation keeps personal notes, project documents, and generated
RAG artifacts on the developer machine by default. Network calls only happen
when the user configures an embedding or generation provider.

## Why Inspectable

RAG quality depends on data and retrieval behavior. RAGentForge exposes chunks,
sources, prompts, traces, and eval reports so users can debug the system rather
than treating it as a black box.

## Why Command-First TUI

The TUI is a Shell for repeated inspection and querying. Commands keep the
workflow explicit, script-like, and easy to document:

```text
/search Agentic RAG
/sources
/source 2
/source next
/sessions failed
/trace
```

It intentionally avoids global single-key shortcuts such as `q`; use `/exit`,
`/quit`, or `/q` from the composer.

## Current v0.2 Boundaries

v0.2 includes lexical, BM25, semantic, and hybrid retrieval plus span-grounded
retrieval evaluation. It does not include reranking, cross-encoder reranking,
LLM-as-judge, answer evaluation, query rewriting, agentic multi-step retrieval,
multi-turn memory as retrieval context, agent tool loops, planning loops,
OCR/scanned PDF support, web UI, vector databases, or TUI write operations such
as ingest/index/eval/config mutation.

The TUI is not a dashboard and does not mutate backend state beyond its local
transcript/session state.

## Future Extension Points

v0.3 is expected to add typed document and project-memory sources and mature
retrieval into inspectable query-processing, candidate retrieval,
deduplication, optional reranking, and context-selection stages. v0.4 can then
build controlled query refinement and iterative retrieval on those explicit
stages. v0.5 is expected to add local comparison views for retrieval and answer
quality. These are extension points, not current v0.2 capabilities.
