# RAGentForge Architecture

> Language: English | [中文](ARCHITECTURE.zh-CN.md)

## Overview

RAGentForge is a local-first, inspectable RAG console. It keeps the current
MVP small: local Markdown/TXT documents are ingested into deterministic chunks,
retrieved with lexical, semantic, or hybrid retrieval, assembled into context,
optionally sent to an OpenAI Responses-compatible generation provider, and
made inspectable through sources, traces, retrieval eval reports, CLI commands,
and a command-first TUI Shell.

## Design Goals

- Keep source documents local and user-owned.
- Store derived state in plain local files under `.ragent/`.
- Make every major RAG step inspectable.
- Keep CLI and TUI behavior backed by shared services.
- Prefer explicit commands over hidden automation.
- Avoid framework lock-in and heavy runtime dependencies in v0.1.

## High-Level Pipeline

```text
local documents
-> ingest
-> deterministic chunks
-> lexical / semantic / hybrid retrieval
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
traces, and eval reports. CLI `ragent ask` is the trace-producing Ask workflow
in v0.1.

### TUI Shell Layer

The Textual Shell in `src/ragent_forge/tui/` is an inspectable command console,
not a management dashboard. It provides a composer, transcript, status line,
command suggestions, selected-source Inspector, read-only summaries, Shell
Search, and Shell Ask.

The Shell intentionally does not run ingest, build indexes, run eval, edit
config, open local files, or add session persistence. Shell Ask does not write
new traces; `/trace` reads traces produced by CLI workflows.

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
- `hybrid` combines lexical and semantic candidates with Reciprocal Rank
  Fusion.

Semantic and hybrid retrieval require a vector index created by
`ragent index build`.

### Generation Services

Generation is optional. With the default `null` provider, Ask stays in
retrieval-only mode and prints retrieved context. When configured with
`openai_responses`, Ask sends a source-grounded prompt to
`{base_url}/responses`.

Chat Completions, streaming, answer evaluation, and LLM-as-judge are not part
of v0.1.

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
/source 2
/sources
/source next
/trace
```

It intentionally avoids global single-key shortcuts such as `q`; use `/exit`,
`/quit`, or `/q` from the composer.

## Current v0.1 Boundaries

v0.1 does not include BM25, reranking, cross-encoder reranking,
LLM-as-judge, answer evaluation, query expansion, multi-turn memory, agent
tool loops, planning loops, PDF/OCR, web UI, vector databases, streaming,
session persistence, or TUI write operations.

The TUI is not a dashboard and does not mutate backend state beyond its local
transcript/session state.

## Future Extension Points

Possible future work includes better retrieval quality, richer source
inspection, answer quality evaluation, controlled agent workflows, and more
demo polish. These are extension points, not current v0.1 capabilities.
