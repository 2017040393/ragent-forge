# RAGentForge Architecture

> 语言: [English](ARCHITECTURE.md) | 中文

## 概览

RAGentForge 是一个本地优先、可检查的 RAG 控制台。它保持当前 MVP 小而清晰：
本地 Markdown/TXT/PDF 文档会被导入为确定性 chunks，再通过 lexical、BM25、
semantic 或 hybrid retrieval 检索，组装为 context，可选发送给 OpenAI
Responses-compatible generation provider，并通过 sources、traces、retrieval
eval reports、CLI commands 和 command-first TUI Shell 变得可检查。

## 设计目标

- 保持源文档本地化，并由用户拥有。
- 将派生状态保存在 `.ragent/` 下的普通本地文件中。
- 让每个主要 RAG 步骤都可检查。
- 让 CLI 和 TUI 行为由共享 services 支撑。
- 偏好显式命令，而不是隐藏自动化。
- 在 v0.2 中避免框架锁定和重型运行时依赖。

## 高层 Pipeline

```text
local documents
-> ingest
-> deterministic chunks
-> normalize query
-> candidate retrieval
-> deduplicate
-> optional rerank
-> context selection
-> context pack
-> optional generation
-> answer + sources
-> traces
-> retrieval eval
-> command-first TUI inspection
```

## 分层

### CLI Layer

`src/ragent_forge/cli/` 中的 CLI package 是主要的可写操作界面。
`cli/__init__.py` 只负责 top-level dispatch 与旧 import facade，`parser.py`
负责参数解析，`cli/handlers/` 分别负责 workspace、chunk、config、trace、index、
retrieval 与 evaluation commands。

CLI workflows 会写入 generations、traces 与 eval reports。Search 和 Ask 都会
持久化 canonical `RetrievalRun` trace payload。

### TUI Shell Layer

`src/ragent_forge/tui/` 中的 Textual Shell 是一个可检查的命令控制台，而不是
管理 dashboard。它提供 composer、transcript、status line、command
suggestions、source/session pickers、selected-answer 和 selected-source
Inspector views、read-only summaries、Shell Search 和 streaming Shell Ask。主
transcript 有意保持 chat-focused：只渲染用户问题和 assistant 回复，并使用
`[1 source]` 或 `[failed]` 这类轻量状态标记；retrieval details 留在 source
pickers、command-result modals 和 Inspector 中。

Shell 有意不运行 ingest、不构建 indexes、不运行 eval、不编辑 config，也不打开
本地文件。它会写入 session artifacts，并让 Search/Ask 按与 CLI 相同的 schema
持久化 operation traces。保存的 Ask turn 只记录对应 `trace_id` 与紧凑 metadata，
需要时可以回读完整 trace；`/trace` 展示任一 surface 写入的 latest trace。Worker
运行时 composer 仍可编辑，并可排队一个 draft；失败时会给出 `/settings`、
`/docs` 或 `/mode bm25` 这类可操作下一步。

### Application Services

`src/ragent_forge/app/services/` 下的 application services 协调用例，例如文件
导入、chunks 列表、config 加载、index 构建、retrieval、ask、trace 写入和
retrieval cases 评估。

这些 services 让 CLI 和 TUI 代码保持薄层，并减少不同 presentation surfaces
之间的重复。

Application services 依赖 ports，而不是具体的文件系统实现或 HTTP library。
顶层 `src/ragent_forge/composition.py` 把 ports 组装到
`src/ragent_forge/infrastructure/` 中的 adapters。`app/composition.py`、
`app/workspace.py` 与 `app/storage.py` 只保留为小型 compatibility facades。

依赖方向是单向的：

```text
CLI / TUI -> application use cases -> core contracts
                         \-> application ports
composition root -> application ports + infrastructure adapters
```

Infrastructure layer 包含本地文件 workspace、原子存储 helpers 和 HTTPX client。
因此 provider 与 workspace implementation 可以在 tests 中替换，也可以在未来
接入 hosted deployment，而不把具体实现泄漏到核心用例中。

CLI 与 TUI 使用同一个 `RetrievalRuntime`，因此 retrieval mode、provider wiring、
prepared cache 和 hybrid configuration 不会在两个 presentation surfaces 之间漂移。

### Retrieval Services

Retrieval 是显式、基于 mode，并通过 `core/retrieval/contracts.py` 中的 typed
contracts 表达的。每次 search run 都会经过并记录以下 stages：

1. query normalization；
2. 通过当前 lexical、BM25、semantic 或 hybrid service 获取 candidates；
3. 按 chunk id 去重；
4. 显式 rerank stage；当前没有配置 reranker 时记录为 skipped；
5. 按请求的 limit 选择 context；
6. 完成 trace。

每次 run 都携带 typed chunk candidates、source coordinates、stage status、inputs、
outputs 和 latency。Search、Ask、TUI search 与 retrieval eval 共用同一条 pipeline，
所以它们的 trace 描述的是同一套 retrieval semantics。

Retrieval modes 本身是：

- `lexical` 使用本地 chunks 上的确定性 token overlap。
- `bm25` 使用预先准备的 term frequency、document frequency 与 document length，
  作为更强的 sparse baseline。
- `semantic` 使用本地 JSONL vector index 上的 cosine similarity。
- `hybrid` 使用 Reciprocal Rank Fusion 融合 BM25 和 semantic candidates。

Semantic 和 hybrid retrieval 需要先通过 `ragent index build` 创建 vector index。

### Generation Services

Generation 是可选的。默认 `null` provider 下，Ask 保持 retrieval-only 模式并
打印检索到的 context。配置为 `openai_responses` 后，Ask 会向
`{base_url}/responses` 发送带来源约束的 prompt。

Chat Completions、answer evaluation 和 LLM-as-judge 不属于当前实现。CLI Ask
path 仍以 trace 为中心；TUI Ask path 可以把 provider deltas 流式写入本地
transcript/session state。

### Workspace Storage

`LocalWorkspace` 统一管理 `.ragent/` 路径并读写派生状态。源文档仍然是事实
来源；workspace files 可以重新生成。

Ingest 先在临时目录写出完整 immutable generation，校验 manifest 与 artifacts，
再原子发布 generation directory，最后才原子替换 `.ragent/current.json`。Index
build 以相同方式发布 child generation。任一写入点失败后，上一代仍保持可读。

当前 workspace schema 是 version 2。Boundary readers 显式执行 v0 -> v1 -> v2
migration，并拒绝未知的未来版本。旧 flat workspace 仍可读取，并可通过
`ragent workspace migrate --dry-run` 检查、通过 `ragent workspace migrate` 升级。

Chunks、ingest summaries、vector records、traces、eval runs 与 sessions 都携带或
解析到 committed snapshot id；reader 会拒绝混合代次。Trace、eval report 与
session 是 immutable generations 之外的 append-oriented artifacts。

重要 workspace files 包括：

```text
.ragent/current.json
.ragent/generations/<snapshot-id>/manifest.json
.ragent/generations/<snapshot-id>/chunks.jsonl
.ragent/generations/<snapshot-id>/ingest_summary.json
.ragent/generations/<snapshot-id>/vector_index.jsonl
.ragent/generations/<snapshot-id>/vector_index_manifest.json
.ragent/config.toml
.ragent/traces/latest_trace.json
.ragent/traces/<trace_id>.json
.ragent/eval/latest_retrieval_eval.json
.ragent/eval/retrieval_eval_<timestamp>.json
.ragent/eval/runs/<run-id>/
.ragent/sessions/latest.json
.ragent/sessions/index.json
.ragent/sessions/session-<id>.json
.ragent/sessions/exports/
```

### Trace and Evaluation

Traces 是本地 JSON 文件，记录精简 workflow metadata。CLI 与 TUI 的 Search/Ask
trace 都嵌入同一个 canonical `RetrievalRun` payload，同时排除完整结果正文。
Retrieval evaluation 读取 JSONL cases，并报告 hit rate、recall、precision、nDCG、
evidence/mapping coverage、overall latency percentiles、stage-level latency
p50/p95、context quality 和 failed cases。

Retrieval eval 只评估检索，不评判生成答案质量。

### 职责与性能边界

Presentation coordination 按 use case 拆分：CLI handlers 位于
`cli/handlers/`，TUI worker/session mapping 位于 `tui/controllers/`。Retrieval
evaluation 的 contracts、JSONL case loader、runner、metrics 与 failure reporting
位于 `app/services/evaluation/`；`retrieval_eval_service.py` 只作为 compatibility
facade。Architecture tests 会锁定这些 module 与 dependency boundaries。

Prepared lexical/BM25 chunks 与 semantic vector records 按 active snapshot id
缓存。Composition root 会跨 TUI runtime builds 保留有界的 per-workspace cache；
没有 snapshot 的 legacy workspace 每次使用 fresh cache。Snapshot 改变时
sparse/dense state 一起失效。Checked-in
`benchmarks/prepared_retrieval_manifest.json` 会分别测一次 cold query 与多次 warm
query，并验证只读取一次 workspace、只准备一次 chunks，以及 warm-cache reuse。
它是 architecture benchmark，不是 v0.3 quality baseline，也不代表 ANN scalability。

## Data Flow

### Ingestion Flow

`ragent ingest <path>` 加载支持的 Markdown/TXT/PDF 文件，确定性切分后发布完整
generation，并写入 ingest trace。

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

TUI Search 和 Ask 会写入与 CLI 对应命令相同的 trace shape。`/trace` 展示 latest
trace 的精简只读摘要；保存的 Ask turn 通过 id 引用对应完整 trace。

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
/sources
/source 2
/source next
/sessions failed
/trace
```

它有意避免 `q` 这类全局单键快捷键；请在 composer 中使用 `/exit`、`/quit`
或 `/q`。

## 当前 v0.2 边界

v0.2 包含 lexical、BM25、semantic 和 hybrid retrieval，以及 span-grounded
retrieval evaluation。它不包含 reranking、cross-encoder reranking、
LLM-as-judge、answer evaluation、query rewriting、agentic multi-step
retrieval、作为 retrieval context 的 multi-turn memory、agent tool loops、
planning loops、OCR/scanned PDF support、web UI、vector databases，也不包含
TUI ingest/index/eval/config mutation 这类写操作。

TUI 不是 dashboard；它只修改本地 session 与 retrieval-trace artifacts。Ingest、
index、eval 与 config mutation 仍属于 CLI 职责。

## 当前 v0.3 基础与未来扩展点

当前架构已经提供 typed retrieval/source contracts、统一且可注入的
`RetrievalEngine`、immutable workspace generations、显式 schema migrations、
snapshot-keyed prepared state、infrastructure adapters、canonical CLI/TUI traces、
trace-linked TUI sessions、聚焦的 presentation/eval modules，以及 cold/warm
benchmark coverage。Reranker 在 v0.3 measurement 证明具体实现有价值之前会保持为
skipped stage。v0.3 可以在同一条 pipeline 上增加 typed project-memory sources；
v0.4 再基于这些 stages 构建受控 query refinement 与 iterative retrieval。
