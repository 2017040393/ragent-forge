# RAGentForge v0.1.0 Release Notes

> 语言: [English](RELEASE_NOTES_V0_1.md) | 中文

## Summary

RAGentForge v0.1.0 是面向 Markdown/TXT 知识库的本地优先、可检查 RAG MVP。
它提供完整本地 loop：ingestion、deterministic chunking、
lexical/semantic/hybrid retrieval、source-grounded Ask、CLI operation traces、
retrieval evaluation 和 command-first TUI inspection。

本文档可以复制到 `v0.1.0` tag 对应的 GitHub Release 中。

## Highlights

- 本地 `.ragent` workspace，保存可检查的派生 artifacts。
- Markdown/TXT ingestion 和 deterministic chunking。
- 不依赖 embeddings 的 lexical retrieval。
- 通过 OpenAI-compatible embedding provider 和本地 JSONL vector index 实现 semantic retrieval。
- 基于 lexical 和 semantic candidates 的 hybrid RRF retrieval。
- 支持可选 OpenAI Responses-compatible generation 的 Ask pipeline。
- 使用默认 `null` generation provider 的 retrieval-only Ask 模式。
- CLI 和 TUI workflows 中的 answer 和 source display。
- CLI ingest、index build、search、ask 和 retrieval eval 的 operation traces。
- 基于 JSONL cases 的 hit@k 和 MRR retrieval evaluation。
- Command-first Textual TUI Shell，包含 inline command suggestions、background
  Ask/Search workers、source navigation 和 selected-source Inspector。

## Included in v0.1.0

- `ragent ingest`：本地 Markdown/TXT ingestion。
- `ragent status`：workspace status。
- `ragent config show` 和 `ragent config init`：本地 provider config。
- `ragent chunks list` 和 `ragent chunks show`：chunk inspection。
- `ragent index build` 和 `ragent index status`：semantic index workflows。
- `ragent search`：支持 `lexical`、`semantic` 和 `hybrid` retrieval modes。
- `ragent ask`：支持 retrieval-only 和 optional generated-answer modes。
- `ragent traces latest`、`ragent traces list` 和 `ragent traces show`。
- `ragent eval retrieval`：支持 hit@1、hit@3、hit@5、requested hit@k、MRR 和
  failed-case reporting。
- `ragent tui`：command-first Textual Shell。

## Command-First TUI

TUI 是单一 Shell interface，包含 transcript、composer、status line、inline
command suggestions 和 Inspector panel。

当前 Shell commands 包括：

```text
/help
/mode lexical|semantic|hybrid
/limit <n>
/context <n>
/prompt on|off
/search <query>
/ask <question>
/sources
/source <rank>
/source next
/source prev
/docs
/trace
/settings
/config
/clear
/exit
/quit
/q
```

普通文本和 `/ask <question>` 会在 background worker 中运行 Shell Ask。
`/search <query>` 会在 background worker 中运行 Shell Search。`/sources` 和
`/source <rank|next|prev>` 会导航 Inspector 中显示的当前 source list。

TUI 读取当前工作目录下默认的 `.ragent` workspace。它不运行 ingest、不构建
indexes、不运行 retrieval eval、不编辑 config，也不打开本地文件。

Shell Ask 在 v0.1 不写入新的 traces。CLI `ragent ask` 仍然是会产生 trace 的
Ask workflow。

## Demo

3-5 分钟 demo script 见 [DEMO_SCRIPT.zh-CN.md](DEMO_SCRIPT.zh-CN.md)。

简短 demo path：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent chunks list --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent traces latest --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
uv run ragent tui
```

## Known Limitations

- Markdown/TXT 是唯一支持的文档格式。
- Lexical retriever 是简单 token overlap，不是 BM25。
- Semantic 和 hybrid retrieval 需要 embedding provider 和已构建的 vector index。
- Vector index 是本地 JSONL，不是生产级 vector database。
- Generation 是可选的，依赖 OpenAI Responses-compatible provider。
- 默认 `null` provider 下，Ask 保持 retrieval-only 模式。
- Retrieval eval 只测量 retrieval behavior；不评估 answer quality。
- Shell Ask 不写入新的 traces。
- TUI 读取当前工作目录下默认的 `.ragent` workspace。

## Non-Goals

v0.1.0 不包含：

- BM25。
- Reranking 或 cross-encoder reranking。
- Answer evaluation。
- LLM-as-judge。
- Query expansion。
- Multi-turn memory。
- Agent loops 或 planning loops。
- PDF 或 OCR support。
- Web UI。
- Chroma、FAISS 或 LanceDB 等 vector databases。
- LangChain 或 LlamaIndex integration。
- OpenTelemetry。
- Streaming。
- Session persistence。
- TUI write operations，例如 ingest、index build、eval 或 config editing。
- Source full-text viewer、local file opening、source table UI 或 mouse source selection。

## Upgrade / Setup Notes

这是早期 v0.1 release。不提供生产迁移路径或 runtime schema migration。

推荐本地 setup：

```bash
uv sync --extra dev
uv run ragent ingest examples/knowledge --workspace .ragent
```

Semantic 和 hybrid retrieval 需要 provider configuration 和 index build：

```bash
uv run ragent config init --workspace .ragent
uv run ragent index build --workspace .ragent
```

API keys 存放在本地 `.ragent/config.toml`；请把该文件视为敏感本地状态。

## Local Release Checklist

- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check .`.
- [ ] Run the demo commands from `docs/DEMO_SCRIPT.md`.
- [ ] Capture TUI screenshots if needed.
- [ ] Create tag `v0.1.0`.
- [ ] Push tag with `git push origin v0.1.0`.
- [ ] Copy these release notes into the GitHub Release.

这个 checklist 只作信息说明。本任务不创建或推送 git tags。

## Suggested Next Versions

未来版本可以改进 retrieval quality，增加更丰富的 source inspection，引入
answer-quality evaluation，并探索一个小型、显式受控的 agent layer。这些是未来
方向，不是 v0.1.0 能力。
