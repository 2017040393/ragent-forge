# E4a Token-budgeted Context Selection

- 日期：2026-07-15
- 状态：accepted-direction
- 实验阶段：pre-v0.3 context-selection experiments
- Parent：[E3b screening conclusion](2026-07-15-e3b-screening-conclusion.md)
- Parent artifacts：[E3b screen result](../benchmarks/results/screens/E3b-pdf-formula-d21c91b)

## 目的

E3b 已通过除 `hybrid_top5_context_tokens` 之外的全部 representation gates。
E4a 不再改变 retrieval ranking，而是隔离 context selection：直接重放冻结的 E3b
Semantic@5 和 Hybrid@5 排名，验证严格 token budget 能否在保留已有相关命中的同时，
把 Hybrid@5 gated context-token ratio 压到 `1.10` 内。

E4a 不重新调用 embedding provider，不修改 document/query representation、chunk、
index、BM25、Hybrid 参数、候选分数或候选顺序。

## Variant

Variant ID：`E4a-ranked-prefix-token-budget`

Selection policy：`ranked_prefix_token_budget_v1`

固定参数：

| Parameter | Value |
|---|---:|
| Candidate depth | 5 |
| Max estimated context tokens | 768 |
| Characters per estimated token | 4 |
| Max context characters | 3072 |

预算沿用 baseline 的 `ceil(chars / 4)` 估算口径。冻结 E0 parent 的 Hybrid@5 gated
平均值为 `725.0238` tokens，`1.10` gate 对应约 `797.5262` tokens；选择常规的
`768`-token 上限，为逐 case 取整和跨 case 平均保留余量。

## Selection Rules

对每个冻结的 E3b Top-5 ranking：

1. 从 rank 1 开始按原顺序处理。
2. 只选择完整 chunk，不截断文本。
3. 如果加入下一个 chunk 会超过 `3072` characters，立即停止。
4. 不跳过超预算 chunk，不从更低 rank 回填，不重排。
5. 最终 selected IDs 必须是原 ranking 的严格前缀。
6. 每个 case 必须至少选择 1 个 chunk，且总估算 token 不得超过 `768`。

该策略只回答“按排名取完整 chunk 前缀时，明确预算是否足够”。如果失败，后续 variant
才能分别研究 skip/backfill、chunk compression 或 evidence-aware packing；这些策略不能
混入 E4a。

## Frozen Inputs

E4a manifest 必须记录并验证：

1. E3b summary 的路径、text-LF SHA-256、variant ID、evaluation commit 和 workspace
   snapshot。
2. E3b `semantic-k5.json` 与 `hybrid-k5.json` 的路径和 SHA-256。
3. 每个 run artifact 的 result fingerprint、16-case 顺序和 mapping coverage。
4. E3b summary 中冻结的 E0 parent Hybrid@5 gated token reference。
5. 当前 Git worktree 必须干净；输出必须写入独立目录。

## Promotion Gates

E4a 只有同时满足以下条件才通过 diagnostic promotion：

1. `semantic_top5_hits_retained`：E3b Semantic@5 已命中的 cases 不得因 selection 丢失。
2. `hybrid_top5_hits_retained`：E3b Hybrid@5 已命中的 cases 不得因 selection 丢失。
3. `hybrid_top5_context_tokens`：gated token ratio `<= 1.10`。
4. `all_contexts_nonempty`：16 cases 在两种 mode 下都至少选择 1 个 chunk。
5. `all_contexts_within_budget`：所有 selected context 都不超过 `768` estimated tokens。
6. `ranked_prefix_preserved`：所有 selected IDs 都是 parent ranking 的严格前缀。

`promoted: true` 只表示 E3b ranking + E4a selection 可以进入 50-case direction
confirmation；它不等于默认启用，也不能作为 release quality 或 answer quality 结论。

## Expected Artifacts

正式运行保存：

- resolved manifest；
- Semantic@5 与 Hybrid@5 的逐 case selection records；
- parent/selected hit、selected IDs、characters、estimated tokens 和 retention；
- aggregated metrics、gate evidence 和最终 decision；
- parent artifact hashes、Git commit 与运行环境。
