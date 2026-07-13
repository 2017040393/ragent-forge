# v0.3 架构收敛方案

> 语言: [English](V0_3_ARCHITECTURE_CONVERGENCE.md) | 中文

## 目的

本文定义 v0.3 retrieval 实验前九项架构风险的实施顺序和验收标准。它们作为一次
连续重构完成，但拆成多个始终保持绿色的 commit。Benchmark 给出证据之前，本轮
不会提前选择具体 reranker 算法或 ANN/vector database。

## 目标依赖方向

```text
CLI / TUI
  -> application use cases
  -> RetrievalEngine 与 application ports
  -> core contracts

composition root
  -> application ports
  -> infrastructure adapters
     - local generation workspace
     - prepared sparse / dense indexes
     - OpenAI-compatible providers
     - trace repository
```

Application 与 core 不得导入 infrastructure。只有 composition root 负责选择具体
adapter。

## Retrieval Runtime

所有 retrieval 入口都使用同一个 `RetrievalEngine.run()` 契约：

```text
strict mode parsing
-> query processing
-> candidate retrieval
-> deduplication
-> optional reranking
-> context selection
-> RetrievalRun persistence
```

每个 stage 都是带 typed input/output 的注入式 protocol。Retrieval mode 只选择
candidate adapter，不再代表一条独立的端到端 pipeline。No-op reranker 可以存在，
但必须显式且可替换。Search、Ask、retrieval eval、comparison、CLI 和 TUI 全部消费
同一种 `RetrievalRun`。

## Typed Data Boundary

Core retrieval data 为以下内容建立 typed models：

- source identity 与 source kind；
- provenance、authority、freshness 和 lifecycle；
- retrieval items 与 candidates；
- stage inputs、outputs、timings 和 failures；
- selected context 与 trace reference。

未验证 dictionaries 只允许出现在 JSON、TOML、HTTP 和 legacy-file 边界。Application
logic 使用前，boundary parser 必须把它们转换成 typed contracts。

## Workspace 事务模型

Workspace generation 一旦提交就保持不可变：

```text
.ragent/
  current.json
  generations/<snapshot-id>/
    manifest.json
    chunks.jsonl
    ingest_summary.json
    vector_index.jsonl
    vector_index_manifest.json
  traces/
  eval/
  sessions/
```

Writer 先在临时目录生成完整 generation，验证后原子 rename 到 `generations/`，最后
原子替换 `current.json`。任何中途失败都必须让上一个 generation 继续可读。Trace、
eval run 和 session 是 append-oriented artifacts，并引用已提交 snapshot id。

旧 flat workspace 继续可读。Migration registry 显式升级已知 schema，并支持 dry
run。未知或未来 schema 给出可操作错误；retrieval mode 与 schema 都不得静默回退。

## Prepared Retrieval State

Prepared state 按 snapshot id 缓存：

- 每个进程和 snapshot 只解析一次 chunks；
- lexical tokenization 只准备一次；
- BM25 document frequencies 和 lengths 只准备一次；
- vector records 与 chunk lookup map 只加载一次；
- snapshot 改变时使全部 prepared state 失效。

当前 corpus 继续支持 exact vector scan，但端口允许以后替换 ANN adapter。Cold 与
warm timings 分开测量；没有 ANN benchmark 时，不声称支持无限规模的次线性查询。

## 统一可观测性

`RetrievalRun` 是 canonical retrieval trace payload。CLI 与 TUI 持久化相同的
operation trace shape。TUI session 只保存 `trace_id` 和紧凑 display metadata，
不再发明第二套 retrieval trace。Evaluation 保存 per-case stage timings，并聚合为
stage-level percentiles。

## 职责拆分

按 use case 而不是任意行数拆分：

- `cli/__init__.py`：只保留 parser facade 与 top-level dispatch；
- `cli/parser.py`：负责参数 parser 构造；
- `cli/handlers/`：ingest、index、retrieval、eval、config 与 trace handlers；
- `tui/main.py`：只保留 Textual composition 与 event routing；
- `tui/controllers/`：Ask、search、session 与 worker coordination；
- retrieval eval：拆为 `evaluation/contracts.py`、`cases.py`、`runner.py`、
  `metrics.py` 与 `reporting.py`；
- infrastructure：filesystem、provider 与 prepared-index adapters。

旧 import path 可以作为薄 compatibility facade 保留，但生产 application code 必须
使用新边界。

## 完成证据

本轮 convergence 已按独立且始终通过验证的 commits 完成。最终的
presentation/evaluation 拆分由 architecture tests 与 compatibility tests 覆盖。
Checked-in benchmark manifest 是 `benchmarks/prepared_retrieval_manifest.json`，可用
以下命令运行：

```text
uv run --extra dev python -m benchmarks.prepared_retrieval
```

Benchmark 会分开报告 cold/warm timings，并验证 cache 的结构性复用；它有意不声称
retrieval quality 已提升，也不声称具备 ANN scalability。Retrieval evaluation report
现在还会在 overall latency metrics 之外，保存 typed stage-level latency summary：
`sample_count`、`average_ms`、`p50_ms` 与 `p95_ms`。

## 验收矩阵

| 风险 | 完成标准 |
| --- | --- |
| Retrieval 仍是黑盒 | 所有入口返回 `RetrievalRun`；stages 可注入并分别测试。 |
| Metadata 类型松散 | Domain/application models 不再使用 `dict[str, Any]`；它只存在于 boundary parsers。 |
| 原子文件不是事务 | 每个 generation 写入点的故障注入都证明旧 snapshot 仍可读取。 |
| Infrastructure 边界概念化 | 架构测试禁止 application/core 导入 infrastructure；provider 和 filesystem code 位于 infrastructure。 |
| 查询重复工作 | Warm query 复用 snapshot-keyed chunks、BM25 state 和 vector records。 |
| CLI/TUI 可观测性不一致 | 两者持久化同一种 retrieval trace，TUI session 引用其 `trace_id`。 |
| 大文件聚集职责 | CLI、TUI、eval 拆成聚焦模块，并保留 compatibility tests。 |
| 非法 mode 变 lexical | 所有边界对非法值给出明确错误。 |
| Schema 没有迁移 | Versioned migration registry 与 legacy golden fixtures 覆盖全部支持升级。 |

## 交付顺序

1. 冻结验收矩阵和架构规则。
2. 引入 typed contracts 与 strict parsing。
3. 让所有 retrieval use case 经过 `RetrievalEngine`。
4. 引入 generation directories 与 schema migrations。
5. 完成 infrastructure 与 composition 边界。
6. 增加 snapshot-keyed prepared retrieval state 和性能断言。
7. 统一 operation traces 与 TUI trace reference。
8. 拆分 presentation/evaluation modules，更新文档并运行完整验证与 benchmark。

每一步都必须在 commit 和 push 之前通过 Pyright、Ruff 与相关测试；最终阶段再次运行
全套验证。
