# Roadmap

> 语言: [English](roadmap.md) | 中文

## v0.1: Local TUI + Inspectable RAG

目标：

- 加载本地 Markdown/TXT 文件。
- 确定性切分文档。
- 支持 lexical、semantic 和 hybrid retrieval。
- 支持带来源的 Ask 和可选 generation。
- 增加本地 traces 和 retrieval evaluation。
- 在 command-first TUI 中显示 sources、traces、settings 和 source inspection。

非目标：

- 真正的 autonomous agents。
- Cloud sync 或 hosted services。
- PDF ingestion 或复杂文档解析。

## v0.2: Retrieval Quality + Better Source Inspection

目标：

- 在当前 token-overlap baseline 之外改进 lexical retrieval quality，把 BM25 作为
  更强的 sparse baseline。
- 增加 retrieval comparison workflows。
- 让 retrieval scores 和 source selection 更容易检查。
- 改进 trace display、export 和 demo polish。

非目标：

- Enterprise search features。
- Multi-user collaboration。
- Provider-specific lock-in。

## v0.3: Retrieval Quality and Efficiency Engineering

核心成果：

- 基于冻结且可复现的 baseline，同时提升 retrieval efficiency、candidate recall
  和 final-result precision。
- 把 query processing、candidate retrieval、deduplication、ranking 和 context
  selection 变成显式、可检查的 stages，但在量化标准确定之前不预设实现方案。
- 为 typed items 提供同一个 retrieval 入口。即使共用 retrieval pipeline，
  document evidence、workspace-local project facts、user notes 和 session memory
  仍保留不同的 provenance 与 lifecycle semantics。
- 保持 workspace-local project memory 可编辑、可审计；它可以成为主要的用户侧
  knowledge surface，但不能抹去 document evidence。
- 扩展 traces 和 evaluation，使每个 retrieval stage 的质量、延迟、context 成本
  和 source behavior 都可以分别诊断。

评测协议：

- 在比较实现方案之前冻结 benchmark manifest，记录 corpus 和 eval set 版本、
  workspace 配置、runtime 与硬件环境、retrieval limits、context budget，以及
  cold/warm run 规则。
- 复用版本化的 v0.2 retrieval eval 作为初始 benchmark，不在对应产品行为出现之前
  预先建立并行的 document-only、memory-only 和 mixed-source suites。保留
  exact-term 与 paraphrase 覆盖，其他 query categories 只在它们成为明确的 v0.3
  行为后加入。
- 使用 v0.2 hybrid retriever 及其当前 document evidence corpus，作为统一 retrieval
  入口的初始 baseline。
- project memory 引入后增加针对性的 memory correctness cases；只有当产品明确支持
  cross-source retrieval、conflict resolution 或 combined context behavior 时，才
  增加 mixed-source cases。
- 每个待测 configuration 至少运行三次，cold 和 warm 结果分开报告，并为每次运行
  持久化 machine-readable artifacts。
- 测量 candidate `hit@k`、`recall@k`；final `precision@k`、`MRR`、`nDCG@k`；
  retrieval latency `p50`、`p95`；candidate 数量；selected context 字符数或 token
  数；以及按 source type 和 pipeline stage 分类的 failures。

初始发布门槛：

- 在 benchmark manifest 和 v0.2 baseline report 提交到仓库之前，不为 v0.3
  选择 retrieval 改进技术。
- quality-oriented configuration 相比 v0.2 hybrid baseline，document
  `recall@20` 和 final `precision@5` 均至少提升 5 个百分点；`MRR` 和 `nDCG@10`
  的回退不得超过 1 个百分点，warm `p95` retrieval latency 不得超过 baseline 的
  1.5 倍，并且平均 selected context tokens 不得增加。
- efficiency-oriented configuration 的 `recall@20`、`precision@5` 和 `nDCG@10`
  与冻结 baseline 的差距不得超过 1 个百分点，同时 warm `p95` retrieval latency
  至少降低 20%，平均 selected context tokens 至少降低 15%。
- 对于 case 数量足以形成稳定比较的已声明 query-category 或 source-kind slice，
  `recall@20` 或 `precision@5` 的回退不得超过 3 个百分点；超出时必须记录明确的
  release exception。
- 推荐的默认配置不能同时在 retrieval quality 和 latency 上严格差于 baseline；
  所有 release-gate 结果都必须能通过提交到仓库的配置和 eval artifacts 复现。
- 首次 baseline run 后可以修订这些目标，但必须在使用 implementation results
  选择技术路线之前冻结，避免根据实验结果移动验收标准。

非目标：

- Hidden long-term memory。
- Cloud profiles。
- 自动导入无关文件。
- Agent-directed iterative 或 autonomous multi-step retrieval。

## v0.4: Minimal Agent Runtime

目标：

- 增加一个小型、受控 runtime，用于显式 multi-step workflows。
- 在 v0.3 的 inspectable retrieval pipeline 上构建 planned query refinement 和
  iterative retrieval。
- 要求可见的 plans、tool steps 和 trace output。
- 保持 side effects 由用户控制。

非目标：

- Fully autonomous background agents。
- Browser automation。
- Distributed task execution。

## v0.5: Evaluation Dashboard

目标：

- 在本地 test sets 上跟踪 retrieval quality 和 answer quality。
- 为 prompts、retrieval stages/settings，以及 document-versus-memory source
  behavior 增加简单 comparison views。
- 支持 learning-oriented experiments。

非目标：

- Enterprise observability。
- Hosted analytics。
- Complex model evaluation infrastructure。

## v0.6: Open-source Polish

目标：

- 改进 documentation 和 examples。
- 稳定 public interfaces。
- 增加 contributor guidance 和 release workflows。

非目标：

- Large marketplace 或 plugin ecosystem。
- Desktop packaging。
- Production SaaS features。
