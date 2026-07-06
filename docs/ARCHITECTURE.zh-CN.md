# RAGentForge Architecture

> 语言: [English](ARCHITECTURE.md) | 中文

## 概览

RAGentForge 是一个本地优先、可检查的 RAG 控制台。它保持当前 MVP 小而清晰：
本地 Markdown/TXT 文档会被导入为确定性 chunks，再通过 lexical、semantic 或
hybrid retrieval 检索，组装为 context，可选发送给 OpenAI
Responses-compatible generation provider，并通过 sources、traces、retrieval
eval reports、CLI commands 和 command-first TUI Shell 变得可检查。

## 设计目标

- 保持源文档本地化，并由用户拥有。
- 将派生状态保存在 `.ragent/` 下的普通本地文件中。
- 让每个主要 RAG 步骤都可检查。
- 让 CLI 和 TUI 行为由共享 services 支撑。
- 偏好显式命令，而不是隐藏自动化。
- 在 v0.1 中避免框架锁定和重型运行时依赖。

## 高层 Pipeline

```text
local documents
-> ingest
-> deterministic chunks
-> lexical / semantic / hybrid retrieval
-> context pack
-> optional generation
-> answer + sources
-> traces
-> retrieval eval
-> command-first TUI inspection
```

## 分层

### CLI Layer

`src/ragent_forge/cli.py` 中的 CLI 是主要的可写操作界面。它运行 ingestion、
status inspection、chunk inspection、config inspection/init、semantic index
build/status、search、ask、trace inspection 和 retrieval evaluation。

CLI workflows 会写入 chunks、summaries、vector indexes、traces 和 eval
reports 等本地 artifacts。v0.1 中，CLI `ragent ask` 是会产生 Ask trace 的
workflow。

### TUI Shell Layer

`src/ragent_forge/tui/` 中的 Textual Shell 是一个可检查的命令控制台，而不是
管理 dashboard。它提供 composer、transcript、status line、command
suggestions、selected-source Inspector、read-only summaries、Shell Search 和
Shell Ask。

Shell 有意不运行 ingest、不构建 indexes、不运行 eval、不编辑 config、不打
开本地文件，也不增加 session persistence。Shell Ask 不写入新的 traces；
`/trace` 读取 CLI workflows 产生的 traces。

### Application Services

`src/ragent_forge/app/services/` 下的 application services 协调用例，例如文件
导入、chunks 列表、config 加载、index 构建、retrieval、ask、trace 写入和
retrieval cases 评估。

这些 services 让 CLI 和 TUI 代码保持薄层，并减少不同 presentation surfaces
之间的重复。

### Retrieval Services

Retrieval 是显式且基于 mode 的：

- `lexical` 使用本地 chunks 上的确定性 token overlap。
- `semantic` 使用本地 JSONL vector index 上的 cosine similarity。
- `hybrid` 使用 Reciprocal Rank Fusion 融合 lexical 和 semantic candidates。

Semantic 和 hybrid retrieval 需要先通过 `ragent index build` 创建 vector index。

### Generation Services

Generation 是可选的。默认 `null` provider 下，Ask 保持 retrieval-only 模式并
打印检索到的 context。配置为 `openai_responses` 后，Ask 会向
`{base_url}/responses` 发送带来源约束的 prompt。

Chat Completions、streaming、answer evaluation 和 LLM-as-judge 不属于 v0.1。

### Workspace Storage

`LocalWorkspace` 统一管理 `.ragent/` 路径并读写派生状态。源文档仍然是事实
来源；workspace files 可以重新生成。

重要 workspace files 包括：

```text
.ragent/chunks/chunks.jsonl
.ragent/ingest/latest_summary.json
.ragent/config.toml
.ragent/index/vector_index.jsonl
.ragent/index/vector_index_manifest.json
.ragent/traces/latest_trace.json
.ragent/traces/<trace_id>.json
.ragent/eval/latest_retrieval_eval.json
.ragent/eval/retrieval_eval_<timestamp>.json
```

### Trace and Evaluation

Traces 是本地 JSON 文件，记录 CLI operations 的精简 metadata 和 workflow
steps。Retrieval evaluation 读取 JSONL cases，并报告 hit@1、hit@3、hit@5、
requested hit@k、MRR 和 failed cases。

Retrieval eval 只评估检索，不评判生成答案质量。

## Data Flow

### Ingestion Flow

`ragent ingest <path>` 加载 Markdown/TXT 文件，跳过不支持的文件，确定性地切
分文档，写入 chunk JSONL、latest ingestion summary 和 ingest trace。

### Search Flow

`ragent search <query>` 从 workspace 读取 chunks，运行选定 retrieval mode，
打印带 source paths 和 previews 的排序结果，并写入 search trace。

缺少 vector index 时，semantic 和 hybrid search 会清晰失败。

### Ask Flow

`ragent ask <question>` 检索 context，构建 context pack，可选构建 generation
prompt，可选调用已配置的 generation provider，打印 answer 或带 sources 的
retrieved context，并写入 Ask retrieval trace。

如果未配置 generation，默认 `null` provider 会让 Ask 保持 retrieval-only 模式。

### Trace Flow

Trace commands 读取本地 trace files：

```text
ragent traces latest
ragent traces list
ragent traces show <trace_id>
```

TUI `/trace` command 会展示 latest trace 的精简只读摘要。

### Evaluation Flow

`ragent eval generate --source <path>` 通过 structured ingestion loader 加载
支持的源文档，抽取稳定 evidence spans，调用已配置的 text generation provider，
并写出 JSONL cases。Markdown 和 TXT 默认启用；text-based PDF extraction 需要
显式加 `--include-pdf`。

这些 generated cases 是 span-based，而不是 chunk-id-based。因此 eval dataset
可以跨 chunk-size 和 chunk-overlap 调整继续复用，同时仍然检查当前 retrieval
system 是否返回覆盖同一份 source evidence 的 chunks。

`ragent eval retrieval --cases <path>` 加载 JSONL cases，运行选定 retrieval
mode，把 span-based cases 映射回当前 workspace chunks，检查 expected chunk ids
或 source paths，写入精简 report，并写入 retrieval eval trace。

Semantic 和 hybrid eval 需要和 semantic/hybrid search 相同的 vector index。

## Workspace Layout

仓库包含小型 demo inputs：

```text
examples/knowledge/
examples/eval/retrieval_cases.jsonl
```

生成的 workspace 默认是：

```text
.ragent/
```

TUI 当前读取当前工作目录下默认的 `.ragent` workspace。启动 TUI 前，请使用
CLI commands 准备该 workspace。

## 为什么 Local-First

Local-first 运行方式默认把个人笔记、项目文档和生成的 RAG artifacts 保存在
开发者机器上。只有当用户配置 embedding 或 generation provider 时，才会发
生网络调用。

## 为什么 Inspectable

RAG 质量取决于数据和检索行为。RAGentForge 暴露 chunks、sources、prompts、
traces 和 eval reports，让用户可以调试系统，而不是把它当作黑盒。

## 为什么 Command-First TUI

TUI 是用于重复检查和查询的 Shell。命令让 workflow 显式、接近脚本，并且
容易写入文档：

```text
/search Agentic RAG
/source 2
/sources
/source next
/trace
```

它有意避免 `q` 这类全局单键快捷键；请在 composer 中使用 `/exit`、`/quit`
或 `/q`。

## 当前 v0.1 边界

v0.1 不包含 BM25、reranking、cross-encoder reranking、LLM-as-judge、answer
evaluation、query expansion、multi-turn memory、agent tool loops、planning
loops、PDF/OCR、web UI、vector databases、streaming、session persistence 或
TUI write operations。

TUI 不是 dashboard，也不会修改后端状态，除了自身本地 transcript/session
state。

## 未来扩展点

未来可能包括更好的 retrieval quality、更丰富的 source inspection、answer
quality evaluation、controlled agent workflows，以及更多 demo polish。这些是
扩展点，不是当前 v0.1 功能。
