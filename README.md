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
- No vector database integration, reranking, or agent workflows yet.
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
- `v0.2`: Retrieval polish and better trace views.
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
- `ragent index build` creates a local JSONL semantic vector index from
  generated chunks using an OpenAI-compatible embeddings provider.
- `ragent index status` reports whether the local semantic index is missing or
  ready.
- `ragent search <query> --retrieval semantic` performs cosine-similarity
  semantic search after an index has been built.
- `ragent search <query> --retrieval hybrid` combines lexical and semantic
  candidates with Reciprocal Rank Fusion after an index has been built.
- Successful `ragent ingest` writes a local JSON trace for the ingest workflow.
- Successful `ragent search` writes a local JSON trace for the search workflow.
- `ragent ask <question>` retrieves local context and can optionally generate an
  answer with an OpenAI Responses-compatible provider.
- `ragent ask <question> --retrieval semantic` uses semantic retrieval before
  assembling the context pack and optional generated answer.
- `ragent ask <question> --retrieval hybrid` uses hybrid RRF retrieval before
  the same context packing and optional generation pipeline.
- `ragent eval retrieval --cases <path>` evaluates retrieval cases from JSONL
  and writes a compact local report.
- `ragent eval retrieval --cases <path> --retrieval hybrid` evaluates hybrid
  RRF retrieval with the same hit-rate and MRR metrics.
- The default `null` generation provider keeps ask in retrieval-only mode when
  real generation is not configured.
- `ragent traces latest`, `ragent traces list`, and
  `ragent traces show <trace_id>` inspect local operation traces.
- `ragent tui` shows Documents workspace status, recent chunk previews, and
  the latest trace summary plus a read-only recent trace history summary.

Reranking, vector databases, and agent workflows are intentionally not
implemented yet.

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

The semantic vector index lives at:

```text
.ragent/index/vector_index.jsonl
.ragent/index/vector_index_manifest.json
```

Retrieval evaluation reports live at:

```text
.ragent/eval/retrieval_eval_<timestamp>.json
.ragent/eval/latest_retrieval_eval.json
```

If the config file is missing, RAGentForge uses the default:

```toml
[generation]
provider = "null"

[embedding]
provider = "none"
```

Official OpenAI Responses API example:

```toml
[generation]
provider = "openai_responses"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key = "sk-..."
timeout_seconds = 60
temperature = 0.2
reasoning_effort = "low"
```

Third-party Responses-compatible API example:

```toml
[generation]
provider = "openai_responses"
base_url = "https://third-party.example.com/v1"
model = "some-responses-compatible-model"
api_key = "tp-..."
timeout_seconds = 60
temperature = 0.2
reasoning_effort = "low"
```

`generation.api_key` stores the provider key directly in the local
`.ragent/config.toml`. Treat this file as sensitive local state. `ragent config
show` hides the key value, and traces do not store the key. The provider calls
`{base_url.rstrip("/")}/responses`. `reasoning_effort` is optional; if omitted,
the model default is used.

OpenAI-compatible embeddings example:

```toml
[embedding]
provider = "openai_embeddings"
base_url = "https://api.openai.com/v1"
model = "text-embedding-3-small"
api_key = "sk-..."
timeout_seconds = 60
batch_size = 64
```

Third-party embeddings-compatible API example:

```toml
[embedding]
provider = "openai_embeddings"
base_url = "https://third-party.example.com/v1"
model = "some-embedding-model"
api_key = "tp-..."
timeout_seconds = 60
batch_size = 64
```

`embedding.api_key` is also stored directly in `.ragent/config.toml`; treat the
file as sensitive. `ragent config show` hides both generation and embedding API
keys. Traces and the vector index do not store API keys. The vector index stores
embedding vectors and metadata but not original full chunk text; full text
remains in `.ragent/chunks/chunks.jsonl`.

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
prints loading/chunking statistics; asking uses lexical retrieval and can
optionally generate an answer when `generation.provider = "openai_responses"`:

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
ragent index status
ragent index build
ragent search "agent memory"
ragent search "agent memory" --retrieval lexical
ragent search "agent memory" --retrieval semantic
ragent search "agent memory" --retrieval hybrid
ragent search "agent memory" --limit 5
ragent eval retrieval --cases eval/retrieval_cases.jsonl
ragent eval retrieval --cases eval/retrieval_cases.jsonl --retrieval lexical
ragent eval retrieval --cases eval/retrieval_cases.jsonl --retrieval semantic
ragent eval retrieval --cases eval/retrieval_cases.jsonl --retrieval hybrid
ragent eval retrieval --cases eval/retrieval_cases.jsonl --limit 5
ragent eval retrieval --cases eval/retrieval_cases.jsonl --report-path report.json
ragent traces latest
ragent traces latest --workspace .ragent
ragent traces list
ragent traces list --limit 20
ragent traces show "<trace_id>"
ragent ask "What is Agentic RAG?"
ragent ask "What is Agentic RAG?" --retrieval lexical
ragent ask "What is Agentic RAG?" --retrieval semantic
ragent ask "What is Agentic RAG?" --retrieval hybrid
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
prints the effective defaults `generation.provider = "null"` and
`embedding.provider = "none"`. `ragent config init` writes that default file, and
`ragent config init --overwrite` replaces an existing config with the default.
`ragent config show` also prints provider-specific generation and embedding
settings while hiding configured API key values. Unsupported provider values
fail clearly.
`ragent chunks list` and `ragent chunks show <chunk_id>` read
`.ragent/chunks/chunks.jsonl` so you can inspect chunking output directly.
`ragent search` defaults to simple
lexical token overlap over `.ragent/chunks/chunks.jsonl`. Retrieval modes are
`lexical`, `semantic`, and `hybrid`; the default remains `lexical`. Semantic
retrieval is opt-in with `--retrieval semantic` and requires `ragent index
build` first. Hybrid retrieval is opt-in with `--retrieval hybrid`, also
requires `ragent index build`, and combines lexical top-N candidates with
semantic top-N candidates using Reciprocal Rank Fusion (RRF).
`ragent index build` embeds chunks in batches and writes
`.ragent/index/vector_index.jsonl`; `ragent index status` reports whether that
index is ready. The MVP computes cosine similarity locally from JSONL vectors.
It does not use BM25, LLM reranking, cross-encoder reranking, score
normalization, FAISS, Chroma, LanceDB, LangChain, LlamaIndex, or a vector
database. Use `ragent chunks show <chunk_id>` to inspect full chunk content.
`ragent ask` defaults to lexical retrieval, assembles a context pack,
and either stays in retrieval-only mode with the default `null` provider or
generates an answer with an OpenAI Responses-compatible provider. `ragent ask
--retrieval semantic` uses the semantic index before the same context packing
and generation step. `ragent ask --retrieval hybrid` uses hybrid RRF retrieval
before the same context packing and generation step. `ragent ask --show-prompt`
shows the actual generation
prompt assembled from the question, retrieved chunks, scores, and source
references; the prompt is only sent when
`generation.provider = "openai_responses"`. Context packs are generated in
memory and are not persisted. `openai_responses` sends a request to
`{base_url}/responses` and supports both the official OpenAI Responses API and
third-party Responses-compatible base URLs. Chat Completions is not implemented
in this step. If no retrieved context is found, `ragent ask` skips generation.
`ragent eval retrieval` reads a user-authored JSONL cases file and checks
whether each query retrieves an expected chunk id or exact source path in the
top-k results. Each line is one case, for example:

```json
{"id":"case-001","query":"How do I configure semantic retrieval?","expected_source_paths":["README.md"]}
```

Each case requires non-empty `id` and `query` fields plus at least one of
`expected_chunk_ids` or `expected_source_paths`. The command reports `hit@1`,
`hit@3`, `hit@5`, requested `hit@k`, and MRR; failed cases are printed in the
CLI. Lexical evaluation is the default and does not require embedding config.
Semantic evaluation uses `--retrieval semantic` and requires `ragent index
build` first. Hybrid evaluation uses `--retrieval hybrid`, requires the same
semantic vector index, and reports `retrieval_method = "hybrid_rrf"` plus RRF
fusion metadata. To compare modes manually, run:

```bash
ragent eval retrieval --cases eval/retrieval_cases.jsonl --retrieval lexical
ragent eval retrieval --cases eval/retrieval_cases.jsonl --retrieval semantic
ragent eval retrieval --cases eval/retrieval_cases.jsonl --retrieval hybrid
```

There is no dedicated compare command in this step. Reports exclude API keys,
full chunk text, embedding vectors, prompts, generated answers, and
answer-quality judgments. This is retrieval evaluation only; answer evaluation,
LLM-as-judge, reranking, charts, dashboards, and TUI eval views are not
implemented yet.
Successful ingest, index build, search, ask retrieval, and retrieval eval
commands write local JSON trace artifacts under `.ragent/traces/<trace_id>.json`;
`.ragent/traces/latest_trace.json` points to the latest operation trace. Current
traces cover ingest, index build, lexical search, semantic search, hybrid
search, ask retrieval, and retrieval eval workflows. Use `ragent traces latest`
for the latest trace, `ragent traces list` for historical trace files, and
`ragent traces show <trace_id>` for one specific trace. No external
observability service is used. The retrieval eval trace operation is
`retrieval_eval` and stores only compact metadata such as case counts, hit
metrics, report path, semantic index metadata, and hybrid RRF metadata when
relevant. The TUI displays the same local
workspace status, a small recent-chunks preview when chunks exist, and a
read-only latest trace summary from `.ragent/traces/latest_trace.json`,
including the latest search or ask retrieval trace after those commands. The
TUI Trace view also shows a read-only recent trace history summary; use
`ragent traces show <trace_id>` for full trace details. Interactive TUI trace
history browsing is not implemented yet.
Default retrieval remains lexical; reranking, vector database integration, and
agent workflows are still not implemented.
