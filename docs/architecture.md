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

## Core Modules

Core modules implement focused RAG capabilities: ingestion, chunking, indexing,
retrieval, generation, and tracing. In this initialization step only the
Markdown/TXT loader, simple chunker, and trace models are real.

## Local Storage Layer

Future versions will store local indexes, traces, and project memory under a
workspace-local `.ragent/` directory. No persistent storage is implemented yet.
