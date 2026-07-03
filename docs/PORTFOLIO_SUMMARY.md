# RAGentForge Portfolio Summary

> Language: English | [中文](PORTFOLIO_SUMMARY.zh-CN.md)

## One-Sentence Summary

RAGentForge is a local-first, inspectable RAG console with deterministic ingestion, lexical/semantic/hybrid retrieval, source-grounded asking, operation tracing, retrieval evaluation, and a command-first Textual TUI with source inspection.

## Short Project Description

RAGentForge is a Python CLI and Textual TUI project for demonstrating an
end-to-end retrieval augmented generation workflow on local Markdown/TXT
knowledge bases. It emphasizes inspectability over abstraction: chunks, vector
indexes, traces, eval reports, answers, and sources are stored or displayed as
local artifacts that can be inspected while debugging or presenting the system.

The current v0.1 scope is intentionally focused. It covers local ingestion,
deterministic chunking, lexical retrieval, optional semantic and hybrid
retrieval with embeddings, source-grounded Ask, CLI traces, retrieval
evaluation, and a command-first TUI for search, ask, and source inspection.

## Chinese Description

RAGentForge 是一个本地优先、可检查的 RAG 控制台，支持本地文档导入、确定性切块、关键词/语义/混合检索、带来源问答、操作追踪、检索评估，以及 command-first TUI 中的来源检查与切换。

## English Description

RAGentForge is a local-first RAG console for developers who want to see and
explain the retrieval pipeline instead of hiding it behind a hosted service or
large framework. It ingests local Markdown/TXT files, writes deterministic
chunks into a `.ragent` workspace, supports explicit lexical, semantic, and
hybrid retrieval modes, and can run source-grounded Ask with either
retrieval-only output or optional OpenAI Responses-compatible generation.

The project also includes CLI operation traces, retrieval evaluation with hit@k
and MRR, and a command-first Textual TUI with background Ask/Search workers,
inline command suggestions, compact source lists, source navigation, and an
Inspector panel.

## Resume Bullets

- Built a local-first RAG console with deterministic ingestion, lexical/semantic/hybrid retrieval, source-grounded asking, operation tracing, and retrieval evaluation.
- Implemented a command-first Textual TUI with background Ask/Search workers, inline command suggestions, source navigation, and an Inspector panel.
- Designed local JSONL workspace storage for chunks, vector index, traces, and retrieval evaluation reports to make RAG workflows inspectable and reproducible.
- Added retrieval evaluation with hit@k and MRR over JSONL cases to measure retrieval behavior before adding heavier ranking or answer-evaluation features.

## Interview Talking Points

- Local-first design keeps documents, derived chunks, traces, and eval artifacts
  inspectable on disk instead of requiring a hosted backend.
- Inspectability matters in RAG because retrieval failures are often data,
  chunking, ranking, or prompt assembly problems rather than model problems.
- Deterministic chunking makes demos, tests, bug reports, and trace comparison
  easier because the same input produces stable chunk identifiers and records.
- Lexical retrieval gives an immediate no-embedding baseline, semantic
  retrieval adds embedding-based matching, and hybrid retrieval combines both
  candidate sets.
- RRF was a practical hybrid MVP because it can combine ranked lexical and
  semantic results without introducing a heavier reranker or trained model.
- The TUI is command-first because RAG debugging often benefits from repeatable
  typed commands, visible transcript history, and explicit source navigation.
- CLI traces and retrieval eval make the project more engineering-oriented:
  users can inspect what happened and measure retrieval behavior before adding
  more complex ranking or answer-evaluation features.
- Future versions could add richer source inspection, better retrieval quality,
  answer-quality evaluation, and a small explicitly controlled agent layer.

## Technical Highlights

- Python CLI built around local application services rather than a hidden
  hosted backend.
- Markdown/TXT ingestion with deterministic chunk records.
- Explicit retrieval modes: `lexical`, `semantic`, and `hybrid`.
- OpenAI-compatible embedding configuration for semantic retrieval.
- Local JSONL vector index for semantic search.
- Hybrid retrieval using Reciprocal Rank Fusion over lexical and semantic
  candidates.
- Ask pipeline with source display and optional OpenAI Responses-compatible
  generation.
- Retrieval-only Ask behavior with the default `null` generation provider.
- CLI traces for operation inspection.
- Retrieval evaluation over JSONL cases with hit@k and MRR.
- Textual TUI with background workers for Ask and Search.

## Architecture Highlights

- `.ragent` workspace keeps generated state local and easy to inspect.
- Presentation layers are split between CLI commands and a Textual TUI.
- Application services provide shared behavior used by both presentation
  layers.
- Core retrieval and generation modules stay focused on pipeline behavior.
- Plain JSONL artifacts are used for chunks, vector index data, traces, and eval
  reports to keep the system reproducible and easy to debug.

## Product / UX Highlights

- Command-first interaction model keeps TUI actions repeatable and easy to
  explain during demos.
- Inline command suggestions help users discover available Shell commands
  without turning the TUI into a dashboard.
- Compact source lists and `/source <rank>`, `/source next`, `/source prev`
  commands make source inspection fast from the composer.
- The Inspector panel keeps the selected source visible while the transcript
  preserves the workflow history.
- Read-only TUI workspace inspection avoids mixing demo-time search/ask flows
  with ingest, index, eval, or config mutation.

## What I Would Improve Next

- Add richer source inspection, such as clearer previews and source metadata,
  while keeping the command-first model.
- Improve retrieval quality with better lexical ranking, optional reranking, or
  query expansion after preserving the current simple baseline.
- Add answer-quality evaluation separately from retrieval evaluation.
- Add more polished release assets such as screenshots, short demo recordings,
  and comparison traces.
- Explore a small explicitly controlled agent layer only after the RAG pipeline
  remains inspectable and measurable.
