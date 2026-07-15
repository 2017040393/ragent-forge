# E4a Context Selection Screening Conclusion

- 日期：2026-07-15
- Variant：`E4a-ranked-prefix-token-budget`
- 状态：promoted-to-direction-confirmation
- Protocol：[Retrieval Representation Screening Protocol](2026-07-14-retrieval-screening-protocol.md)
- Specification：[E4a token-budgeted context selection](2026-07-15-e4a-token-budget-context-selection.md)
- Parent：[E3b screening conclusion](2026-07-15-e3b-screening-conclusion.md)
- Artifacts：[E4a screen result](../benchmarks/results/context-screens/E4a-ranked-prefix-token-budget-ed8d609)

## 实验边界

E4a 以冻结的 E3b Semantic@5 和 Hybrid@5 run artifacts 为 parent，完全离线重放
已有 ranking。它没有重新调用 embedding provider，也没有改变 E3b 的 document/query
representation、index、chunk、候选分数或候选顺序。

唯一改变是 context selection：按 parent ranking 的原顺序取完整 chunk 前缀；如果
下一个 chunk 超过预算就停止，不截断、不跳过、不回填。预算为 `768` estimated tokens，
按 `ceil(characters / 4)` 估算，即最多 `3072` characters。

Parent artifacts 的 summary/run hashes、E3b evaluation commit、workspace snapshot、
result fingerprints、16-case 顺序和 mapping coverage 均经过验证。E4a 没有产生新的
embedding 或 index。

## 结果

| Configuration | Parent hits retained | Average selected tokens | Gated token ratio |
|---|---:|---:|---:|
| Semantic@5 | 11/11 | 677.3125 | n/a |
| Hybrid@5 | 10/10 | 697.3125 | 0.9683 |

Hybrid@5 gated average 为 `702.0714` tokens，相对 E0 parent 的 `725.0238` reference
为 `0.9683`，低于 `1.10` gate。所有 32 个 mode/case combinations 都至少保留一个
chunk，最大单 case 为 `758` estimated tokens，所有 selected IDs 都是 parent ranking
的严格前缀。

## Decision

六个 E4a gates 全部通过：

- Semantic@5 parent hits retained：`11/11`；
- Hybrid@5 parent hits retained：`10/10`；
- Hybrid@5 context-token ratio：`0.9683 <= 1.10`；
- contexts nonempty：`0` violations；
- budget respected：`0` violations；
- ranked prefix preserved：`0` violations。

因此 E4a `valid: true`、`promoted: true`，E3b ranking 加 E4a selection 可以进入
50-case direction confirmation。这个 promotion 只说明固定 screen 上的 context
selection gate 已修复，不代表默认启用或 release quality 已确认。

## Interpretation

E3b 的问题不是所有 Top-5 排名都错误，而是固定 Top-5 中完整 chunk 的总长度过高。
E4a 在不改变排名的情况下，把平均上下文压到 gate 以内，同时保留 parent 已有的
Semantic/Hybrid Top-5 相关命中。这支持“先固定 ranking，再用独立 context budget 处理
长度偏差”的架构方向。

下一阶段是 50-case direction confirmation，继续冻结 E3b workspace、E2 instructed
query、E3b ranking 和 E4a `768`-token prefix policy；在该确认完成前，不应把此组合
写入默认配置。
