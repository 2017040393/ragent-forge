# RAGentForge v0.1 Demo Script

> 语言: [English](DEMO_SCRIPT.md) | 中文

## Demo 目标

把 RAGentForge 展示为一个本地优先、可检查的 RAG 控制台。在 3-5 分钟内，demo
应证明本地 Markdown 文件可以被导入、切块、检索、带来源 Ask、trace、评估，
并通过 command-first TUI Shell 检查。

Demo 要保持诚实：semantic 和 hybrid retrieval 需要配置 embedding provider
并构建 vector index；默认 `null` provider 下 generation 可能是禁用的；Shell
Ask 在 v0.1 不会写入新的 traces。

## 30 秒项目介绍

RAGentForge 是一个本地优先的 RAG 控制台，用来检查完整 retrieval workflow。
它将派生状态存储在 `.ragent` 中，支持确定性切块、lexical、semantic 和
hybrid retrieval、可选 source-grounded generation、operation traces、
retrieval evaluation，以及 command-first Textual TUI。

核心想法是 inspectability。项目不把 RAG pipeline 隐藏在托管应用或框架抽象
后面，而是把 chunks、sources、traces、eval reports 和 TUI transcript state
作为本地 artifacts 暴露出来。

## Demo Setup

从仓库根目录运行命令。

如有需要，先安装依赖：

```bash
uv sync --extra dev
```

从干净 demo workspace 开始：

```bash
rm -rf .ragent
```

Windows PowerShell 中使用：

```powershell
Remove-Item -Recurse -Force .ragent
```

## Demo Flow

### 1. 清理 Workspace

说明 `.ragent` 是派生本地状态，可以重新生成。

```bash
rm -rf .ragent
uv run ragent status --workspace .ragent
```

预期讲法：源文档仍在 `examples/knowledge`，但派生 workspace 还没有准备好。

### 2. 导入本地知识

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
```

指出 ingestion 会写入 chunks、ingest summary，以及 `.ragent` 下的 CLI
operation trace。

### 3. 检查 Chunks

```bash
uv run ragent chunks list --workspace .ragent
```

如果时间允许，可选运行：

```bash
uv run ragent chunks show "<chunk_id>" --workspace .ragent
```

说明 deterministic chunk ids 和 JSONL storage 让 pipeline 更容易 debug 和 test。

### 4. 运行 Lexical Search

```bash
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

说明 lexical search 在 ingestion 后立即可用，适合精确术语、文件名、配置字段
和快速本地 demo。

### 5. Ask with Sources

```bash
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

如果 generation 仍使用默认 `null` provider，说明这是预期行为：Ask 保持
retrieval-only 模式，并显示 retrieved context 和 sources。如果配置了
`openai_responses`，该命令可以生成 source-grounded answer。

### 6. 检查 Trace

```bash
uv run ragent traces latest --workspace .ragent
```

说明 CLI workflows 会写入 traces。这很重要，因为它让每次 operation 后的系统
都可检查。Shell Ask 在 v0.1 不写入新的 traces；CLI `ragent ask` 是会产生
trace 的 Ask workflow。

### 7. 运行 Retrieval Evaluation

```bash
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
```

指出 hit@k 和 MRR。说明这是 retrieval-only evaluation，不是 answer quality
evaluation，也不是 LLM-as-judge。

### 8. 启动 TUI Shell

```bash
uv run ragent tui
```

说明 TUI 读取当前工作目录下默认的 `.ragent` workspace。v0.1 中它不接受
`--workspace` 参数。

### 9. 在 TUI 中 Search

在 composer 中输入：

```text
/help
/search Agentic RAG
```

指出 inline command suggestions、background Shell Search、transcript 中的
source lists，以及 selected-source Inspector。

### 10. Navigate Sources

输入：

```text
/source 2
/sources
/source next
/source prev
```

说明 source navigation 是 command-first。v0.1 中没有 source table UI 或 mouse
selection。

### 11. 展示只读 Workspace 状态

输入：

```text
What is Agentic RAG?
/trace
/settings
/exit
```

说明普通文本会在 background worker 中运行 Shell Ask。`/trace` 和 `/settings`
是只读 inspection commands。使用 composer 中的 `/exit`、`/quit` 或 `/q` 退出；
没有 `q` 这样的全局单键快捷键。

## Demo 时可以怎么说

- “源文档保持本地；`.ragent` 保存派生 artifacts。”
- “Chunking 是确定性的，因此失败更容易复现。”
- “Retrieval modes 是显式的：lexical、semantic 和 hybrid。”
- “Semantic 和 hybrid retrieval 需要 embeddings 和已构建的 vector index。”
- “Ask 可以不依赖 generation。默认 `null` provider 下，它保持 retrieval-only 模式。”
- “CLI workflows 写入 traces。TUI 读取并显示这些 traces。”
- “TUI 有意设计成 command-first，而不是 dashboard-first。”
- “这不是自主 Agent 框架，而是一个小型、可检查的 RAG MVP。”

## 技术上展示了什么

- `.ragent` 下的 local workspace design。
- Markdown/TXT ingestion 和 deterministic chunking。
- 不依赖 embeddings 的 lexical retrieval。
- Semantic 和 hybrid retrieval 作为可选 indexed modes。
- Source-grounded Ask 和 retrieval-only fallback behavior。
- 用于 debug 和解释的 CLI operation traces。
- 基于 JSONL cases 的 hit@k 和 MRR retrieval evaluation。
- CLI 和 TUI 共享 application services。
- 用于 non-blocking Ask/Search 的 Textual Shell workers。
- TUI 中的 command suggestions、source navigation 和 selected-source inspection。

## 未配置 Embeddings 时的 Fallback Path

保持 demo 为 lexical-only：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
uv run ragent tui
```

在 TUI 中使用：

```text
/mode lexical
/search Agentic RAG
What is Agentic RAG?
/sources
/source next
/trace
/exit
```

明确说明 semantic 和 hybrid modes 是支持的，但需要 embedding provider 和
`uv run ragent index build --workspace .ragent`。

## Closing Summary

RAGentForge v0.1 展示了完整本地 RAG loop：ingest、chunk、search、ask、trace、
evaluate，并通过 command-first TUI inspect。它有意避免 production claims 和
autonomous agent behavior。项目价值在于让 RAG pipeline 可见、可复现，并且适合
在面试或作品集 review 中讨论。
