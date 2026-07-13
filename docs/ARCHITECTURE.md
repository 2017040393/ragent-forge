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
-> normalize query
-> candidate retrieval
-> deduplicate
-> optional rerank
-> context selection
-> context pack
-> optional generation
-> answer + sources
-> traces
-> retrieval eval
-> command-first TUI inspection
```

## Layers

### CLI Layer

The CLI package in `src/ragent_forge/cli/` is the primary write-capable
operator surface. `cli/__init__.py` contains only top-level dispatch and the
historical import facade, `parser.py` owns argument parsing, and
`cli/handlers/` owns workspace, chunk, config, trace, index, retrieval, and
evaluation commands.

CLI workflows write local artifacts such as generations, traces, and eval
reports. Search and Ask persist the canonical `RetrievalRun` trace payload.

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
config, or open local files. It writes local session artifacts and persists
Search/Ask operation traces with the same schema as the CLI. Saved Ask turns
store the corresponding `trace_id`, so compact transcript metadata can resolve
to the full trace. `/trace` reads the latest trace from either surface. Running
workers leave the composer editable and can queue one draft; worker failures
surface actionable next steps such as `/settings`, `/docs`, or `/mode bm25`.

### Application Services

Application services under `src/ragent_forge/app/services/` coordinate use
cases such as ingesting files, listing chunks, loading config, building the
index, running retrieval, asking questions, writing traces, and evaluating
retrieval cases.

These services keep CLI and TUI code thin and reduce duplication between
presentation surfaces.

Application services depend on ports rather than on a concrete filesystem or
HTTP library. The top-level composition root wires those ports to adapters in
`src/ragent_forge/infrastructure/`. `app/composition.py`, `app/workspace.py`,
and `app/storage.py` remain small compatibility import facades.

The dependency direction is one-way:

```text
CLI / TUI -> application use cases -> core contracts
                         \-> application ports
composition root -> application ports + infrastructure adapters
```

The infrastructure layer contains the local filesystem workspace, atomic
storage helpers, and the HTTPX client. This keeps provider and workspace
implementations replaceable in tests and in future hosted deployments.

### Composition and Dependency Rules

Concrete service construction lives in `src/ragent_forge/composition.py`. CLI
and TUI use the same `RetrievalRuntime`, so retrieval modes, provider wiring,
prepared caches, and hybrid configuration cannot drift between surfaces.

The dependency direction is one-way:

```text
CLI / TUI -> application -> core
composition -> application ports <- infrastructure adapters
```

Core modules do not import `app`, `cli`, or `tui`. Application services depend
on narrow protocols such as `ChunkReader`, `ConfigWorkspace`, and
`VectorIndexWorkspace` where they only need a subset of workspace behavior.
The architecture tests enforce these import boundaries.

### Retrieval Services

Retrieval is explicit, mode-based, and represented by typed contracts in
`core/retrieval/contracts.py`. A search run is observable as these stages:

1. query normalization;
2. candidate retrieval through the selected lexical, BM25, semantic, or hybrid
   service;
3. candidate deduplication by chunk id;
4. an explicit rerank stage, currently recorded as skipped when no reranker is
   configured;
5. context selection to the requested limit;
6. trace finalization.

Each run carries typed chunk candidates, source coordinates, stage status,
inputs, outputs, and latency. Search, Ask, TUI search, and retrieval eval share
the same pipeline, so their traces describe the same retrieval semantics.

The retrieval modes themselves are:

- `lexical` uses deterministic token overlap over local chunks.
- `bm25` uses prepared term frequencies, document frequencies, and document
  lengths as the stronger sparse baseline.
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

Ingest writes a complete immutable generation in a temporary directory,
validates its manifest and artifacts, atomically publishes the directory, and
atomically replaces `.ragent/current.json` last. Index builds publish a child
generation in the same way. A failure at any write point leaves the previous
generation readable.

The current workspace schema is version 2. Boundary readers run explicit v0 to
v1 to v2 migrations and reject unknown future versions. Legacy flat workspaces
remain readable and can be inspected or upgraded with
`ragent workspace migrate --dry-run` and `ragent workspace migrate`.

Chunks, ingest summaries, vector records, traces, eval runs, and sessions carry
or resolve a committed snapshot id. Readers reject mixed generations. Traces,
eval reports, and sessions are append-oriented artifacts outside immutable
generation directories.

Important workspace files include:

```text
.ragent/current.json
.ragent/generations/<snapshot-id>/manifest.json
.ragent/generations/<snapshot-id>/chunks.jsonl
.ragent/generations/<snapshot-id>/ingest_summary.json
.ragent/generations/<snapshot-id>/vector_index.jsonl
.ragent/generations/<snapshot-id>/vector_index_manifest.json
.ragent/config.toml
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

Traces are local JSON files that record compact workflow metadata. CLI and TUI
Search/Ask traces embed the same canonical `RetrievalRun` payload without full
result text. Retrieval evaluation reads JSONL cases and reports hit rate,
recall, precision, nDCG, evidence and mapping coverage, overall latency
percentiles, stage-level latency p50/p95, context quality, and failed cases.

Retrieval eval is retrieval-only. It does not judge generated answer quality.

### Responsibility and Performance Boundaries

Presentation coordination is split by use case: CLI handlers live under
`cli/handlers/`, while TUI worker and session mapping live under
`tui/controllers/`. Retrieval evaluation keeps its contracts, JSONL case
loader, runner, metrics, and failure reporting under
`app/services/evaluation/`; `retrieval_eval_service.py` is a compatibility
facade. Architecture tests lock these module and dependency boundaries.

Prepared lexical/BM25 chunks and semantic vector records are cached by active
snapshot id. The composition root keeps a bounded per-workspace cache across
TUI runtime builds; legacy workspaces without a snapshot receive a fresh cache.
A snapshot change invalidates both sparse and dense state. The checked-in
`benchmarks/prepared_retrieval_manifest.json` separates one cold query from
repeated warm queries and verifies one workspace read, one prepared chunk load,
and warm-cache reuse. It is an architecture benchmark, not a v0.3 quality
baseline or an ANN scalability claim.

## Data Flow

### Ingestion Flow

`ragent ingest <path>` loads supported Markdown/TXT/PDF files, chunks them
deterministically, publishes a complete generation, and writes an ingest trace.

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

TUI Search and Ask write the same trace shape as their CLI counterparts. The
TUI `/trace` command shows a compact read-only summary of the latest trace, and
saved Ask turns reference the exact trace by id.

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

The TUI is not a dashboard. It mutates only local session and retrieval-trace
artifacts; ingest, index, eval, and config mutation remain CLI responsibilities.

## Current v0.3 Foundation and Future Extension Points

The current architecture provides typed retrieval/source contracts, one
injected `RetrievalEngine`, immutable workspace generations, explicit schema
migrations, snapshot-keyed prepared state, infrastructure adapters, canonical
CLI/TUI traces, trace-linked TUI sessions, focused presentation/eval modules,
and cold/warm benchmark coverage. The reranker deliberately remains a skipped
stage until v0.3 measurements justify an implementation. v0.3 can add typed
project-memory sources on the same pipeline; v0.4 can build controlled query
refinement and iterative retrieval on these explicit stages.
