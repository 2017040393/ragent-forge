# RAGentForge

A local-first TUI workbench for inspectable Agentic RAG workflows over personal knowledge bases.

RAGentForge is an early-stage Python project for developers who want to query,
organize, and reason over their own Markdown/TXT notes, project documents,
learning materials, and interview preparation files. Early versions focus on
local, inspectable RAG workflows rather than full autonomous agents.

## Project Goals

- Provide a TUI-first workspace for local knowledge exploration.
- Keep ingestion, chunking, retrieval, generation, and traces inspectable.
- Start with Markdown and TXT files before expanding to richer formats.
- Use Python to iterate quickly on AI system design.
- Leave room for future Rust modules only after real bottlenecks appear.

## Early Non-Goals

- No cloud service, hosted backend, or enterprise knowledge base.
- No Web UI, desktop UI, authentication, or multi-user workflow.
- No complete RAG pipeline, real LLM calls, embeddings, or vector database yet.
- No no-code platform, complex agent autonomy, plugin system, or distributed jobs.
- No Rust, PyO3, native extensions, or mixed-language build system in this step.

## Core Principles

1. Local-first before cloud.
2. TUI-first before Web/Desktop UI.
3. Inspectable before autonomous.
4. Python-first for fast AI system iteration.
5. Rust-ready only as a future option for performance-critical modules.

## Target Environment

- Python 3.11+
- Local developer machines
- Markdown/TXT personal knowledge bases
- Terminal-first workflows

## Planned Roadmap

- `v0.1`: Local TUI + inspectable RAG foundations.
- `v0.2`: Hybrid retrieval and better trace views.
- `v0.3`: Project memory over local workspaces.
- `v0.4`: Minimal agent runtime with explicit controls.
- `v0.5`: Evaluation dashboard for retrieval and answer quality.
- `v0.6`: Open-source polish, examples, and contributor ergonomics.

## Current Status

This repository currently contains the project skeleton, documentation, typed
data models, Markdown/TXT loader, simple chunker, trace model, and a minimal
Textual TUI skeleton.

Implemented so far:

- `ragent ingest` scans local files or folders for Markdown/TXT documents.
- Ingestion loads supported documents and skips unsupported files.
- Loaded documents are chunked with the deterministic `SimpleChunker`.
- `IngestResult` returns document, chunk, skipped-file, and chunking statistics.
- The CLI writes chunks and the latest ingestion summary under `.ragent/`.
- `ragent status` reads `.ragent/` and reports whether the local workspace is
  ready, incomplete, or not initialized.
- `ragent config show` and `ragent config init` inspect or write optional local
  generation config.
- `ragent chunks list` and `ragent chunks show <chunk_id>` inspect generated
  chunks from the local workspace.
- `ragent search <query>` performs deterministic lexical search over generated
  chunks.
- Successful `ragent ingest` writes a local JSON trace for the ingest workflow.
- Successful `ragent search` writes a local JSON trace for the search workflow.
- `ragent ask <question>` previews retrieved context without generating answers.
- A null generation provider interface records that generation is not configured.
- `ragent traces latest`, `ragent traces list`, and
  `ragent traces show <trace_id>` inspect local operation traces.
- `ragent tui` shows Documents workspace status, recent chunk previews, and
  the latest trace summary plus a read-only recent trace history summary.

Real embeddings, vector database integration, answer generation, persistent
retrieval indexes, and agent workflows are intentionally not implemented yet.

## Local Workspace

RAGentForge stores generated local state under `.ragent/`.

The knowledge base directory contains human-readable source documents. The
`.ragent/` directory contains derived system data such as chunks, ingestion
summaries, traces, memory, and future indexes.

The source documents are the source of truth; `.ragent/` is derived and can be
regenerated.

`ragent status` reads workspace state from:

```text
.ragent/chunks/chunks.jsonl
.ragent/ingest/latest_summary.json
```

Each successful traced operation writes a trace file under:

```text
.ragent/traces/<trace_id>.json
```

`ragent traces latest` reads the latest local trace pointer from:

```text
.ragent/traces/latest_trace.json
```

Optional generation config lives at:

```text
.ragent/config.toml
```

If the config file is missing, RAGentForge uses the default:

```toml
[generation]
provider = "null"
```

The TUI Documents view reads the same workspace files. The TUI Trace view reads
`.ragent/traces/latest_trace.json` and recent trace files under
`.ragent/traces/<trace_id>.json`. TUI trace history is read-only and not
interactive yet; use `ragent traces show <trace_id>` to inspect a specific trace.
TUI trace selection and TUI ingestion interactions are not implemented yet.

## Development Setup

With `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

With standard Python tooling on macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

With standard Python tooling on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

## Basic Usage

These commands are available now. Ingestion scans local Markdown/TXT files and
prints loading/chunking statistics; asking previews retrieved context without
generating an answer:

```bash
ragent --help
ragent tui
ragent ingest examples/knowledge
ragent ingest examples/knowledge --workspace .ragent
ragent status
ragent status --workspace .ragent
ragent config show
ragent config init
ragent config init --overwrite
ragent chunks list
ragent chunks list --limit 20
ragent chunks show "<chunk_id>"
ragent search "agent memory"
ragent search "agent memory" --limit 5
ragent traces latest
ragent traces latest --workspace .ragent
ragent traces list
ragent traces list --limit 20
ragent traces show "<trace_id>"
ragent ask "What is Agentic RAG?"
ragent ask "What is Agentic RAG?" --limit 5
ragent ask "What is Agentic RAG?" --show-prompt
ragent ask "What is Agentic RAG?" --show-prompt --limit 5
```

`ragent tui` launches the minimal Textual application. `ragent ingest` loads and
chunks local Markdown/TXT files without creating embeddings or a vector index.
It writes `.ragent/chunks/chunks.jsonl` and
`.ragent/ingest/latest_summary.json` by default.
`ragent status` reports whether those workspace files are present and readable.
`.ragent/config.toml` is optional; when it is missing, `ragent config show`
prints the effective default `generation.provider = "null"`. `ragent config init`
writes that default file, and `ragent config init --overwrite` replaces an
existing config with the default. Only the `null` generation provider is supported
right now. Unsupported provider values fail clearly, and no API keys are read.
`ragent chunks list` and `ragent chunks show <chunk_id>` read
`.ragent/chunks/chunks.jsonl` so you can inspect chunking output before
broader retrieval work is implemented. `ragent search` also reads
`.ragent/chunks/chunks.jsonl` and uses simple lexical token overlap to rank
chunks. It is not semantic search and does not use embeddings, vector
databases, BM25, reranking, LLMs, or answer generation. Use
`ragent chunks show <chunk_id>` to inspect full chunk content. `ragent ask`
runs in retrieval-only mode using the same lexical search, displays retrieved
context, and clearly skips answer generation. `ragent ask --show-prompt` also
shows a deterministic local prompt preview assembled from the question,
retrieved chunks, and source references; the preview is not sent to any LLM.
Context packs are generated in memory and are not persisted. Prompt previews
are local and are not sent anywhere. `ragent ask` builds a generation request
and sends it to the current null generation provider, which returns
`not_configured` with no answer. No real LLM provider is implemented yet, no API
keys are read, and future providers such as OpenAI or Ollama can be added later
behind the generation provider interface. Successful ingest, search, and ask
retrieval commands write local JSON trace artifacts under
`.ragent/traces/<trace_id>.json`; `.ragent/traces/latest_trace.json` points to
the latest operation trace. Current traces cover ingest, lexical search, and ask
retrieval workflows. Use `ragent traces latest` for the latest trace,
`ragent traces list` for historical trace files, and
`ragent traces show <trace_id>` for one specific trace. No external observability
service is used, and this does not implement semantic, vector, LLM, or agent
retrieval tracing. The TUI displays the same local workspace status, a small
recent-chunks preview when chunks exist, and a read-only latest trace summary
from `.ragent/traces/latest_trace.json`, including the latest search or ask
retrieval trace after those commands. The TUI Trace view also shows a read-only
recent trace history summary; use `ragent traces show <trace_id>` for full trace
details. Interactive TUI trace history browsing is not implemented yet.
`ragent ask` does not generate or fake an answer yet.
