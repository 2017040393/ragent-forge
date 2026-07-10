# RAGentForge Portfolio Summary

> 语言: [English](PORTFOLIO_SUMMARY.md) | 中文

## One-Sentence Summary

RAGentForge 是一个本地优先、可检查的 RAG 控制台，支持 structured
Markdown/TXT/PDF ingestion、lexical/BM25/semantic/hybrid retrieval、
source-grounded asking、span-grounded evaluation、deterministic failure
analysis、retrieval compare，以及带 streaming Ask、saved sessions、聚焦的
source/session pickers 和 answer-bound source inspection 的 command-first
Textual TUI。

## Short Project Description

RAGentForge 是一个 Python CLI 和 Textual TUI 项目，用来在本地
Markdown/TXT/PDF 知识库上展示端到端 retrieval augmented generation workflow。它强调
inspectability，而不是抽象隐藏：chunks、vector indexes、traces、eval
reports、answers 和 sources 都会作为本地 artifacts 被保存或展示，便于调试或
演示系统。

当前 v0.2 surface 有意保持聚焦。它覆盖 Markdown/TXT/PDF 结构化导入、确定性切块、
lexical 和 BM25 retrieval、带 embeddings 的可选 semantic 和 hybrid retrieval、
source-grounded Ask、CLI traces、span-grounded eval generation、
evidence-to-current-chunk mapping、持久化 eval reports、deterministic failure
analysis、retrieval compare，以及用于 search、ask 和 source/session picker
inspection 的 command-first TUI。当前 TUI 还支持 streaming Ask、saved sessions、
session export、filtered session review 和 selected-answer review。

## Chinese Description

RAGentForge 是一个本地优先、可检查的 RAG 控制台，支持 Markdown/TXT/PDF 结构化导入、确定性切块、lexical/BM25/semantic/hybrid 检索、带来源问答、操作追踪、span-grounded eval generation、evidence-to-current-chunk mapping、deterministic failure analysis、retrieval compare，以及 command-first TUI 中的 streaming Ask、saved sessions、session export 和绑定到具体回答的来源检查。

## English Description

RAGentForge is a local-first RAG console for developers who want to see and
explain the retrieval pipeline instead of hiding it behind a hosted service or
large framework. It ingests local Markdown/TXT/PDF files through a structured
document pipeline, writes deterministic chunks into a `.ragent` workspace,
supports explicit lexical, BM25, semantic, and hybrid retrieval modes, and can
run source-grounded Ask with either retrieval-only output or optional OpenAI
Responses-compatible generation.

The project also includes CLI operation traces, retrieval evaluation with
Hit@k、Recall@k、MRR、latency 和 context-size metrics, 不绑定当前 chunk ids 的
span-based synthetic eval generation, evaluation-time
evidence-to-current-chunk mapping, persisted eval reports, deterministic
failure analysis, retrieval compare across modes，以及 command-first Textual
TUI with background Ask/Search workers, contextual inline command suggestions,
streaming answer display, clean chat transcript badges, focused source/session
pickers, saved-session management, export/branch/rerun helpers, queued drafts,
actionable worker failures, source navigation, and an Inspector panel tied to
the selected answer.

## Resume Bullets

- Built a local-first RAG console with structured Markdown/TXT/PDF ingestion, lexical/BM25/semantic/hybrid retrieval, source-grounded asking, operation tracing, and retrieval evaluation.
- Implemented a command-first Textual TUI with background Ask/Search workers, streaming answer display, saved sessions, inline command suggestions, filtered session pickers, source navigation, queued drafts, actionable failures, and an Inspector panel tied to the selected answer.
- Designed local JSONL workspace storage for chunks, vector index, traces, and persisted retrieval evaluation reports to make RAG workflows inspectable and reproducible.
- Added retrieval evaluation with Hit@k, Recall@k, MRR, latency, and context-size metrics over JSONL cases.
- Added span-based synthetic eval generation, evidence-to-current-chunk mapping, deterministic failure analysis, and retrieval comparison so datasets can be reused across chunking, retrieval, and ranking experiments.

## Interview Talking Points

- Local-first design 将 documents、derived chunks、traces 和 eval artifacts 保持
  在磁盘上可检查，而不是依赖托管 backend。
- Inspectability 对 RAG 很重要，因为 retrieval failures 往往来自数据、切块、
  排序或 prompt assembly，而不只是模型问题。
- Deterministic chunking 让 demos、tests、bug reports 和 trace comparison 更容易，
  因为相同输入会产生稳定的 chunk identifiers 和 records。
- Lexical retrieval 提供不需要 embeddings 的即时 baseline；BM25 加强 sparse
  keyword matching；semantic retrieval 增加 embedding-based matching；hybrid
  retrieval 结合 BM25 和 semantic candidates。
- RRF 是实用的 hybrid MVP，因为它可以融合 BM25 和 semantic ranked results，而
  不引入更重的 reranker 或训练模型。
- TUI 采用 command-first，因为 RAG debugging 需要可重复 typed commands、可见的
  transcript history 和显式 source navigation。
- CLI traces 和 retrieval eval 让项目更工程化：用户可以检查发生了什么，并在
  增加更复杂 ranking 或 answer-evaluation features 前测量 retrieval behavior。
- Span-grounded eval 通过在 evaluation 时把 evidence spans 映射到当前 chunk store，
  让生成数据集可以跨 chunking changes 复用。
- Deterministic failure analysis 让 miss 可以被稳定复查，而不引入 LLM-as-judge
  或不可复现 scoring。
- v0.3 预计把 project memory 与 inspectable retrieval quality improvements
  结合；v0.4 增加受控的 multi-step retrieval 和 agent workflows；v0.5 增加本地
  retrieval 与 answer-quality comparison views。

## Technical Highlights

- 围绕本地 application services 构建的 Python CLI，而不是隐藏托管 backend。
- Markdown/TXT/PDF structured ingestion 和 deterministic chunk records。
- PDF ingestion 支持 page text、table extraction、page ranges 和 source metadata。
- 显式 retrieval modes：`lexical`、`BM25`、`semantic` 和 `hybrid`。
- 面向 semantic retrieval 的 OpenAI-compatible embedding configuration。
- 用于 semantic search 的本地 JSONL vector index。
- 使用 Reciprocal Rank Fusion 融合 BM25 和 semantic candidates 的 hybrid retrieval。
- 带 source display 和可选 OpenAI Responses-compatible generation 的 Ask pipeline。
- 默认 `null` generation provider 下的 retrieval-only Ask behavior。
- 用于 operation inspection 的 CLI traces。
- 基于 JSONL cases 的 Hit@k、Recall@k、MRR、latency 和 context-size metrics
  retrieval evaluation。
- Span-based generated eval cases 将测试数据集和当前 chunk store 解耦，便于比较
  chunking 与 retrieval strategies。
- Span-grounded eval cases 的 evidence-to-current-chunk mapping。
- Deterministic failure analysis 和 compact failure reports。
- 持久化 eval reports，包含 summary JSON/Markdown、compact cases JSONL 和
  failures JSONL。
- 跨 lexical、BM25、semantic 和 hybrid modes 的 retrieval compare。
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
- Focused source pickers 加上 `/source <rank>`、`/source next`、`/source prev`
  commands，让用户可以快速从 composer 检查 sources。
- Inspector panel 在主 transcript 保持 user/assistant chat 的同时，让
  selected-answer 和 selected-source details 保持可见。
- Read-only TUI workspace inspection 避免把 demo 时的 search/ask flows 和 ingest、index、
  eval 或 config mutation 混在一起。

## What I Would Improve Next

- 增加更丰富的 source inspection，例如更清晰的 previews 和 source metadata，同时保持 command-first model。
- 把 document evidence 和 project memory 作为 typed retrieval sources，然后通过
  inspectable query processing、optional reranking、query expansion 和
  source-aware evaluation 提升 single-pass retrieval quality。
- 增加与 retrieval evaluation 分离的 answer-quality evaluation。
- 增加 short demo recordings 和更大一些的 benchmark-style corpora。
- 只有在 v0.3 retrieval pipeline 保持可检查、可测量后，再探索受控的
  multi-step retrieval 和小型 agent layer。
