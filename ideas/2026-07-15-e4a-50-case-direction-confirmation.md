# E4a 50-case Direction Confirmation

- 日期：2026-07-15
- 状态：completed-not-confirmed
- Candidate：`E3b-pdf-formula + E4a-ranked-prefix-token-budget`
- Parent：[Post-architecture baseline](2026-07-14-pre-v0-3-post-architecture-baseline.md)
- Screen result：[E4a screening conclusion](2026-07-15-e4a-screening-conclusion.md)
- Result：[E4a 50-case conclusion](2026-07-15-e4a-50-case-direction-confirmation-conclusion.md)
- Artifacts：[E4a-50-case-8299040](../benchmarks/results/direction-confirmations/E4a-50-case-8299040)

## 目的

16-case screen 已表明 E3b ranking 加 E4a context selection 能改善数学 PDF 检索并
通过 context-token gate。50-case direction confirmation 使用完整冻结数据集验证该方向，
确认 screen gains 不是小样本选择造成的，同时继续隔离 ranking 和 context selection。

本阶段不是正式 release baseline，不做三轮 latency 或 ranking-stability claim。只有通过
本协议后，候选才有资格进入正式多轮确认或默认策略评审。

## Frozen Inputs

1. Dataset：`examples/eval/v0_2_retrieval_baseline.generated.jsonl`，50 cases。
2. Corpus：4 documents、1744 chunks，chunk boundaries 与 E0-E4a 完全相同。
3. Parent：冻结的 post-architecture E0 三轮 baseline summary 和 Semantic/Hybrid@5
   三轮 artifacts。
4. Candidate workspace：E3b snapshot
   `snapshot-20260715T070507Z-e2ed57b0`。
5. Document representation：`cleaned_pdf_formula_text_v1`。
6. Query representation：`instructed_query_v1`。
7. Embedding model：`Qwen/Qwen3-VL-Embedding-8B`，4096 dimensions。
8. Ranking modes：Semantic@5、Hybrid@5；Hybrid 参数保持冻结值。
9. Context selection：`ranked_prefix_token_budget_v1`，`768` estimated tokens，
   `4` characters per token。

Chunk-content fingerprint、index-input fingerprint、dataset/corpus hashes、workspace build
commit、parent summary/run hashes 和 query-cache lineage 必须写入 manifest 并在运行前
验证。

## Query Cache

E4a screen 的 instructed-query cache 只有 16 个 selected cases。正式确认以该 cache 为
只读 seed：

1. 已有 16 个 query vectors 必须命中且内容不变。
2. 缺失的 34 个完整集 queries 只生成一次并写入本轮独立 cache。
3. Semantic@5 完成 cache 扩充，Hybrid@5 必须 50/50 全部命中。
4. 最终 cache 保存 provider、model、query representation、dimension、source hash 和
   50-entry fingerprint。

不允许覆盖 E2/E3/E4a 的历史 cache。

## Execution

每种 mode 只执行一次完整 ranking：

```text
50 cases
-> frozen E3b retrieval runtime
-> Top-5 ranking artifact
-> offline E4a ranked-prefix selection
-> selection artifact and gate report
```

Ranking quality 使用未裁剪的 E3b Top-5；context metrics 使用 E4a selected prefix。
Selector 不重新运行 retrieval，不改变 candidate scores、order 或 IDs。

## Confirmation Gates

相对冻结 E0 parent mean：

1. `semantic_hit_direction`：Semantic@5 hit-rate delta `>= 0.04`，即至少净增 2/50。
2. `hybrid_hit_nonnegative`：Hybrid@5 hit-rate delta `>= 0.00`。
3. `no_new_missed_source`：两种 mode 合计新增 `missed_source` 为 0；parent 三轮从未出现
   `missed_source` 的 case 不得新增该失败。
4. `semantic_context_hits_retained`：E3b Semantic@5 ranking hits 经 E4a selection 后
   loss 为 0。
5. `hybrid_context_hits_retained`：E3b Hybrid@5 ranking hits 经 E4a selection 后
   loss 为 0。
6. `hybrid_context_tokens`：E4a Hybrid@5 average selected-context token ratio 相对 E0
   parent mean `<= 1.10`。
7. `context_selection_invariants`：100 个 mode/case combinations 全部 nonempty、预算内，
   selected IDs 为 parent ranking 严格前缀，mapping coverage 为 1.0。

同时记录 parent consensus transitions：一个 case 在 E0 三轮中至少两次命中视为 parent
consensus hit。Gains、losses、failure-type transitions 和 selection count distribution
全部保存，但不新增事后 gate。

## Decision Rule

- `valid: true` 要求所有输入、cache、workspace、artifact 和 Git provenance 校验通过。
- `confirmed: true` 要求 `valid: true` 且七个 gates 全部通过。
- 未通过时保留完整 artifacts，并判断失败属于 ranking direction、source recall 还是
  context selection。
- 通过只表示方向得到完整 50-case 支持；在正式多轮确认或默认策略评审前，不修改默认
  representation、retrieval 参数或 context budget。

## Expected Artifacts

- resolved manifest；
- 50-entry instructed-query cache；
- Semantic@5、Hybrid@5 ranking artifacts，以及绑定 ranking、query cache SHA 的
  resume checkpoints；
- 两种 mode 的 E4a selection artifacts；
- parent consensus transitions、aggregate metrics、七个 gates 和 summary；
- 独立结论文档与 reproduction command。

## Recorded Result

本协议已在 commit `8299040a04ae32f5d5d632713d45f14d6759e9c8` 上执行。
运行 `valid: true`、`confirmed: false`：Semantic/Hybrid ranking gates 通过，但 768-token
E4a prefix 丢失 4 个 Hybrid Top-5 hits，未通过 `hybrid_context_hits_retained`。
完整指标、失败 cases 和 budget frontier 见结论文档；不得将先前 16-case promotion 解释为
full-dataset confirmation。
