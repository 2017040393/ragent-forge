# Architecture

RAGentForge uses a small layered architecture. The TUI and CLI are presentation
surfaces; core RAG behavior belongs below the application service layer.

```text
TUI Layer
  |
Application Services
  |
Core Modules
  |-- Ingestion
  |-- Chunking
  |-- Indexing
  |-- Retrieval
  |-- Generation
  |-- Tracing
  |
Local Workspace
  |-- knowledge/
  |-- .ragent/chunks/
  |-- .ragent/ingest/
  |-- .ragent/index/
  |-- .ragent/traces/
  |-- .ragent/memory/
```

## TUI Layer

The Textual interface owns layout, navigation, and display state. It must not
contain core RAG logic. User actions should call application services and render
their results.

## Application Services

Services coordinate use cases such as ingesting files, asking questions, and
creating traces. They provide stable entry points for both the CLI and TUI.
`LocalWorkspace` handles reading and writing workspace state so presentation
layers do not parse `.ragent/` files directly.

## Core Modules

Core modules implement focused RAG capabilities: ingestion, chunking, indexing,
retrieval, generation, and tracing. In this initialization step only the
Markdown/TXT loader, simple chunker, and trace models are real.

## Local Storage Layer

RAGentForge stores generated local state under a workspace-local `.ragent/`
directory. Source documents remain the source of truth; `.ragent/` contains
derived system data and can be regenerated.

The current ingestion flow writes:

- `.ragent/chunks/chunks.jsonl` for generated `DocumentChunk` records.
- `.ragent/ingest/latest_summary.json` for the latest ingestion summary.

The status flow reads those same files to report whether the local workspace is
ready, incomplete, or not initialized.

Future versions may add local indexes, traces, memory, and evaluation artifacts
under the same workspace directory.
