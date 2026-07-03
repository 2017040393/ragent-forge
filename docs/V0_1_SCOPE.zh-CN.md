# RAGentForge v0.1 Scope

> 语言: [English](V0_1_SCOPE.md) | 中文

## v0.1 目标

RAGentForge v0.1 是一个本地优先、可检查的 RAG MVP。它的目标是让基础 RAG
workflow 易于理解并适合 demo：导入本地文档，确定性地切块，检索相关来源，
可选生成答案，检查 traces，运行 retrieval eval，并通过 command-first TUI
Shell 探索结果。

## 已包含能力

- 本地 Markdown/TXT 文档导入。
- 确定性切块。
- 本地 `.ragent/` workspace storage。
- JSONL chunk storage。
- Lexical retrieval。
- OpenAI-compatible semantic embeddings。
- 本地 JSONL vector index。
- Semantic retrieval。
- Hybrid RRF retrieval。
- 可选 OpenAI Responses-compatible generation。
- 使用默认 `null` provider 的 retrieval-only Ask 模式。
- Answer sources。
- CLI operation traces。
- 使用 hit@k 和 MRR 的 retrieval evaluation。
- Command-first Textual TUI Shell。
- Inline Shell command suggestions。
- 使用 `/sources`、`/source <rank>`、`/source next` 和 `/source prev` 的
  Shell source navigation。
- Selected-source Inspector。

## 明确非目标

v0.1 有意不包含：

- BM25。
- Reranking。
- Cross-encoder reranking。
- LLM-as-judge。
- Answer evaluation。
- Query expansion。
- Multi-turn memory。
- Agent tool loops。
- Planning loops。
- PDF support。
- OCR。
- Web UI。
- Vector databases。
- LangChain。
- LlamaIndex。
- Chroma、FAISS 或 LanceDB。
- OpenTelemetry。
- Streaming。
- Session persistence。
- TUI ingest execution。
- TUI index build execution。
- TUI eval execution。
- TUI config editing。
- Source full-text viewer。
- Local file opening。
- Mouse source selection。

## 为什么这不只是 Toy Demo

- Workflow 写入可检查的本地 artifacts，而不是把状态隐藏在内存里。
- Chunks、vector indexes、traces 和 eval reports 都是普通本地文件。
- Retrieval modes 是显式且可测试的。
- Hybrid retrieval 记录 fusion metadata。
- Ask 可以不依赖 generation 运行，使 retrieval 行为可见。
- CLI workflows 和 TUI 共享 application services，而不是重复 backend logic。
- TUI 是真正的 command shell，具备 worker-backed Ask/Search、suggestions、
  source navigation 和 Inspector。
- Retrieval eval 在 JSONL cases 上提供可重复的 hit-rate 和 MRR 检查。

## 已知限制

- Markdown/TXT 是唯一支持的文档格式。
- Lexical retriever 是简单 token overlap，不是 BM25。
- Semantic 和 hybrid retrieval 需要配置 embedding provider 并构建 vector index。
- Vector index 是本地 JSONL，不是生产级 vector database。
- Generation 是可选的，依赖 OpenAI Responses-compatible provider。
- Retrieval eval 不评估生成答案质量。
- Shell Ask 不写入新的 traces；CLI `ragent ask` 仍是会产生 Ask trace 的 workflow。
- TUI 是可检查的 shell，不是完整管理 dashboard。

## v0.1 Readiness Checklist

- [x] Ingest local documents。
- [x] Inspect chunks。
- [x] Lexical search。
- [x] Semantic search。
- [x] Hybrid search。
- [x] Ask with sources。
- [x] Trace CLI Ask。
- [x] Retrieval eval。
- [x] Command-first TUI。
- [x] Source navigation。

## 建议未来 Roadmap

下面的 roadmap 是未来工作，不是当前能力。

### v0.2 Retrieval Quality

- 改进 lexical retrieval quality。
- 增加 retrieval comparison workflows。
- 改进 source ranking inspection。
- 只有在当前 baseline 被测量后，才考虑 BM25 或 reranking。

### v0.3 Answer Quality and Evaluation

- 增加 answer-quality evaluation。
- 增加 prompt comparison workflows。
- 跟踪 groundedness 和 citation quality。
- 保持 answer eval 与 retrieval eval 分离。

### v0.4 Agent Layer

- 增加小型、显式、用户可控的 agent layer。
- 要求可见的 plans、tool steps 和 traces。
- 保持 side effects opt-in 且可检查。
- 避免隐藏的 autonomous loops。
