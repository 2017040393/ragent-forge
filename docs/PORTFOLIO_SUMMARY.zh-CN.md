# RAGentForge Portfolio Summary

> 语言: [English](PORTFOLIO_SUMMARY.md) | 中文

## One-Sentence Summary

RAGentForge 是一个本地优先、可检查的 RAG 控制台，支持 deterministic
ingestion、lexical/semantic/hybrid retrieval、source-grounded asking、
operation tracing、retrieval evaluation，以及带 source inspection 的
command-first Textual TUI。

## Short Project Description

RAGentForge 是一个 Python CLI 和 Textual TUI 项目，用来在本地 Markdown/TXT
知识库上展示端到端 retrieval augmented generation workflow。它强调
inspectability，而不是抽象隐藏：chunks、vector indexes、traces、eval
reports、answers 和 sources 都会作为本地 artifacts 被保存或展示，便于调试或
演示系统。

当前 v0.1 范围有意保持聚焦。它覆盖本地导入、确定性切块、lexical retrieval、
带 embeddings 的可选 semantic 和 hybrid retrieval、source-grounded Ask、CLI
traces、retrieval evaluation，以及用于 search、ask 和 source inspection 的
command-first TUI。

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

- Local-first design 将 documents、derived chunks、traces 和 eval artifacts 保持
  在磁盘上可检查，而不是依赖托管 backend。
- Inspectability 对 RAG 很重要，因为 retrieval failures 往往来自数据、切块、
  排序或 prompt assembly，而不只是模型问题。
- Deterministic chunking 让 demos、tests、bug reports 和 trace comparison 更容易，
  因为相同输入会产生稳定的 chunk identifiers 和 records。
- Lexical retrieval 提供不需要 embeddings 的即时 baseline；semantic retrieval
  增加 embedding-based matching；hybrid retrieval 结合两类 candidates。
- RRF 是实用的 hybrid MVP，因为它可以融合 lexical 和 semantic ranked results，
  而不引入更重的 reranker 或训练模型。
- TUI 采用 command-first，因为 RAG debugging 需要可重复 typed commands、可见的
  transcript history 和显式 source navigation。
- CLI traces 和 retrieval eval 让项目更工程化：用户可以检查发生了什么，并在
  增加更复杂 ranking 或 answer-evaluation features 前测量 retrieval behavior。
- 未来版本可增加更丰富的 source inspection、更好的 retrieval quality、
  answer-quality evaluation，以及小型、显式受控的 agent layer。

## Technical Highlights

- 围绕本地 application services 构建的 Python CLI，而不是隐藏托管 backend。
- Markdown/TXT ingestion 和 deterministic chunk records。
- 显式 retrieval modes：`lexical`、`semantic` 和 `hybrid`。
- 面向 semantic retrieval 的 OpenAI-compatible embedding configuration。
- 用于 semantic search 的本地 JSONL vector index。
- 使用 Reciprocal Rank Fusion 融合 lexical 和 semantic candidates 的 hybrid retrieval。
- 带 source display 和可选 OpenAI Responses-compatible generation 的 Ask pipeline。
- 默认 `null` generation provider 下的 retrieval-only Ask behavior。
- 用于 operation inspection 的 CLI traces。
- 基于 JSONL cases 的 hit@k 和 MRR retrieval evaluation。
- 使用 background workers 运行 Ask 和 Search 的 Textual TUI。

## Architecture Highlights

- `.ragent` workspace 让生成状态保持本地且易于检查。
- Presentation layers 分为 CLI commands 和 Textual TUI。
- Application services 提供两个 presentation layers 共享的行为。
- Core retrieval 和 generation modules 聚焦 pipeline behavior。
- Chunks、vector index data、traces 和 eval reports 使用普通 JSONL artifacts，
  让系统可复现且易于 debug。

## Product / UX Highlights

- Command-first interaction model 让 TUI actions 可重复，并且适合 demo 讲解。
- Inline command suggestions 帮助用户发现可用 Shell commands，而不把 TUI 变成 dashboard。
- Compact source lists 和 `/source <rank>`、`/source next`、`/source prev` commands
  让用户可以快速从 composer 检查 sources。
- Inspector panel 在 transcript 保留 workflow history 的同时，让 selected source 保持可见。
- Read-only TUI workspace inspection 避免把 demo 时的 search/ask flows 和 ingest、index、
  eval 或 config mutation 混在一起。

## What I Would Improve Next

- 增加更丰富的 source inspection，例如更清晰的 previews 和 source metadata，同时保持 command-first model。
- 在保留当前简单 baseline 的前提下，通过更好的 lexical ranking、optional reranking 或 query expansion 改进 retrieval quality。
- 增加与 retrieval evaluation 分离的 answer-quality evaluation。
- 增加更完善的 release assets，例如 screenshots、short demo recordings 和 comparison traces。
- 只有在 RAG pipeline 仍保持可检查、可测量后，再探索小型、显式受控的 agent layer。
