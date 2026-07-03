# RAGentForge Project Walkthrough

> 语言: [English](PROJECT_WALKTHROUGH.md) | 中文

## 这个 Demo 展示什么

这个 walkthrough 展示当前本地 RAG loop：

```text
ingest local documents
-> inspect chunks
-> run lexical retrieval
-> optionally build a semantic index
-> run semantic or hybrid retrieval
-> ask with sources
-> inspect traces
-> run retrieval evaluation
-> inspect results in the command-first TUI
```

Demo 使用仓库中的 `examples/knowledge` 和 `examples/eval` 文件。

## 前置条件

- Python 3.11 或更新版本。
- 已安装 `uv`。
- 从仓库根目录运行命令。
- 可选：在 `.ragent/config.toml` 中配置 embedding provider，用于 semantic 和
  hybrid retrieval。
- 可选：在 `.ragent/config.toml` 中配置 generation provider，用于生成答案。

安装依赖：

```bash
uv sync --extra dev
```

## Step 1: 准备 Workspace

从本地 workspace 目录开始：

```bash
uv run ragent status --workspace .ragent
uv run ragent config show --workspace .ragent
```

如果想先创建 config file 再编辑 provider settings：

```bash
uv run ragent config init --workspace .ragent
```

默认 config 使用 `generation.provider = "null"` 和 `embedding.provider = "none"`。

## Step 2: 导入本地知识

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
```

这会加载 Markdown/TXT 文件，创建确定性 chunks，写入
`.ragent/chunks/chunks.jsonl`，写入 ingestion summary，并写入 CLI operation
trace。

## Step 3: 检查 Workspace 状态

```bash
uv run ragent status --workspace .ragent
```

关注 `Status: ready`、document count、chunk count，以及 chunk 和 summary 文件
路径。

## Step 4: 检查 Chunks

```bash
uv run ragent chunks list --workspace .ragent
uv run ragent chunks show "<chunk_id>" --workspace .ragent
```

运行 demo 时，从 `chunks list` 复制一个 chunk id 到 `chunks show`。这能证明
chunking output 是可检查的。

## Step 5: 运行 Lexical Search

```bash
uv run ragent search "What is RAG?" --retrieval lexical --workspace .ragent
```

Lexical retrieval 是默认模式，不需要 embeddings 或 vector index。

## Step 6: 可选 Semantic Index Build

Semantic 和 hybrid retrieval 需要 embedding provider 和 vector index。在
`.ragent/config.toml` 中配置 `[embedding]` 后运行：

```bash
uv run ragent index build --workspace .ragent
uv run ragent index status --workspace .ragent
```

Index 会以 JSONL 加 manifest 的形式保存在本地 `.ragent/index/` 下。

## Step 7: 运行 Semantic 和 Hybrid Search

Index 存在后：

```bash
uv run ragent search "What is Agentic RAG?" --retrieval semantic --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

Semantic search 使用本地 vector similarity。Hybrid search 使用 Reciprocal
Rank Fusion 融合 lexical 和 semantic candidates。

## Step 8: Ask 一个问题

Lexical Ask 不需要 index：

```bash
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

Hybrid Ask 需要 vector index：

```bash
uv run ragent ask "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

默认 `null` generation provider 下，Ask 保持 retrieval-only 模式。当配置了
`generation.provider = "openai_responses"` 后，CLI Ask 可以生成带来源约束的
答案。

如果想检查 prompt 且不隐藏 retrieval context：

```bash
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --show-prompt --workspace .ragent
```

CLI `ragent ask` 会写入 Ask trace。TUI 中的 Shell Ask 在 v0.1 不写入新的
traces。

## Step 9: 在 TUI 中检查 Sources

准备好 `.ragent` 后，从仓库根目录启动 TUI：

```bash
uv run ragent tui
```

当前 TUI command 不接受 `--workspace` 参数；它读取当前工作目录下默认的
`.ragent` workspace。

试试这个 Shell sequence：

```text
/help
/search Agentic RAG
/source 2
/sources
/source next
/source prev
What is Agentic RAG?
/trace
/settings
/exit
```

`/sources` 显示当前 source list。`/source <rank>`、`/source next` 和
`/source prev` 会切换 Inspector 中显示的 source。

TUI 有意避免 `q` 这种全局单键快捷键；请在 composer 中使用 `/exit`、`/quit`
或 `/q`。

## Step 10: 检查 Traces

```bash
uv run ragent traces latest --workspace .ragent
uv run ragent traces list --workspace .ragent
uv run ragent traces show "<trace_id>" --workspace .ragent
```

用 `traces list` 找到 trace id，再传给 `traces show`。CLI ingest、index build、
search、ask 和 retrieval eval workflows 会写入 traces。

## Step 11: 运行 Retrieval Evaluation

Lexical eval：

```bash
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
```

Semantic 和 hybrid eval 需要 vector index：

```bash
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval semantic --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval hybrid --workspace .ragent
```

Retrieval eval 会报告 hit@1、hit@3、hit@5、requested hit@k、MRR 和 failed
cases。它不评估 answer quality，也不运行 LLM-as-judge。

## 观察重点

- 源文档保留在 `examples/knowledge` 下。
- 派生 artifacts 保留在 `.ragent` 下。
- Chunks 是可读的 JSONL records。
- Search 和 Ask 输出包含 source paths 和 chunk ids。
- Semantic 和 hybrid modes 在 vector index 存在前会清晰失败。
- CLI Ask 写入 traces；Shell Ask 不写入。
- TUI 中的 `/trace` 读取 latest existing CLI trace。
- Retrieval eval 使用 `examples/eval` 下的小型 JSONL cases。

## Troubleshooting

- `No chunks found`：运行 `uv run ragent ingest examples/knowledge --workspace .ragent`。
- `vector index not found`：配置 embeddings，然后运行
  `uv run ragent index build --workspace .ragent`。
- `generation.provider = null`：generated answers 已禁用；Ask 保持
  retrieval-only 模式。
- TUI 没有 trace：先运行 CLI workflow，例如 `ingest`、`search`、`ask` 或
  `eval retrieval`。
- Eval 没命中 expected paths：检查 `uv run ragent chunks list --workspace .ragent`，
  确保 case file 使用 exact source paths 或 repo-relative suffixes，例如
  `examples/knowledge/rag_basics.md`。

## Cleanup

`.ragent/` 目录包含派生本地状态。要从头重跑 demo，删除 `.ragent/` 并重新运
行 ingest command。
