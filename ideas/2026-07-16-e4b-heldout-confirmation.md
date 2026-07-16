# E4b Held-out Confirmation

- 日期：2026-07-16
- 状态：accepted-protocol
- Candidate：`E4b1-ranked-query-fragments`
- Frozen implementation：`ccc930ae27740fca1da706c08487cc66cbb5493f`
- Development result：[E4b development conclusion](2026-07-16-e4b-fragment-packing-development-conclusion.md)

## 目的与隔离边界

E4b1 在已观察的 canonical 50 cases 上通过全部 development gates，但该数据不能再次
作为独立 confirmation。本阶段从同一冻结 corpus 中选择此前从未进入任何 canonical
case 的 PDF evidence spans，生成新的 query/reference answer，并在不修改 E4b1 参数的
前提下确认 fragment evidence preservation。

Held-out 只确认 context selection，不重新声明 E3b+E2 ranking 相对 E0 的质量提升，也不
产生 release 或 answer-quality claim。E4b1 implementation、Top-5 depth、768-token budget、
window scoring、water-filling、tokenization、evidence n-gram 和所有 gates 在生成 held-out
queries 前冻结。

## Span Selection

Canonical dataset 使用过 25 个 unique spans。Held-out 从两本 PDF 的剩余 spans 中选择，
排除正文少于 1000 characters 的 span。每个 source 独立执行 deterministic farthest-fill：

1. occupied indexes 初始化为 canonical manifest 的 `selected_span_indexes`；
2. 每轮选择与所有 occupied indexes 的最小距离最大的 eligible span；
3. 距离相同时选择较小 span index；
4. 选中后加入 occupied，直到每本 PDF 得到 5 spans。

冻结 spans：

| Source | Span index | Page | Text SHA-256 |
|---|---:|---:|---|
| High-Dimensional Probability | 0 | 2 | `0237992663ad6b58b8e8dafe4d91a17aebfd2f6ed6144e67af4a16fed9bf9457` |
| High-Dimensional Probability | 296 | 298 | `426526402b17bf2c884dd2568735711f379757a2c4a3a73159cd1bc8c065d47c` |
| High-Dimensional Probability | 211 | 215 | `b20f417dc01a329a2fe02c8f97443be9a6eaeea10202a31634b1ab722f315c27a` |
| High-Dimensional Probability | 248 | 251 | `6577c2cf7807f0005a0970442c8821af1e84f2ab2e20a4ae8c575623375092fa4` |
| High-Dimensional Probability | 15 | 20 | `88f6dd6ea8b42503b206ce50359b60ef0963c440426668ab27f824f99f6bef492` |
| Linear Algebra Done Right | 408 | 410 | `44d7920dd18508be5ef80108ebdd41bc1ddfde1d959af155044845400f7764ca2` |
| Linear Algebra Done Right | 2 | 6 | `c1bbf4abec2fd94151fbfeefe9615fc78599183d91362456ef08466aaf519bbc1` |
| Linear Algebra Done Right | 341 | 343 | `bab783cd648d3940510c446faf06aa7b263cc14001249519e5160199951377d72` |
| Linear Algebra Done Right | 387 | 389 | `a16266a5ff75faa211df7020d24ae613dbfc50a8e155c218075a423dd9919e89f` |
| Linear Algebra Done Right | 21 | 26 | `5a0151541a55a26517e768840b81bf993ecaf56a2b0dc663c7ce37999ff0268e0` |

Selector 必须重新提取 spans 并验证 index、page、character count、text hash 和 canonical
exclusion。任何 mismatch 都中止生成，禁止按当前位置选择替代 span。

## Dataset Generation

- 10 frozen spans x 2 questions = 20 cases；
- case IDs：`e4b-heldout-000001` 到 `e4b-heldout-000020`；
- generation service：现有 `EvalDatasetGenerationService`；
- generation provider/model 与 canonical dataset 相同：
  `openai_responses` / `gpt-5.6-luna`；
- reasoning effort `medium`、temperature `0.2`；
- 每个 case 只包含一个 frozen evidence span；
- source paths 与 span IDs 写成 repository-relative POSIX paths；
- dataset、selection manifest、generation config 和每个 span hash 全部归档。

生成后只做结构与人工质量检查：JSONL 可加载、20 unique nonempty queries、reference
answer 被 evidence 支持、无明显答案泄漏、无 canonical query duplication。不得根据 E4b
retrieval 结果删除或替换困难 case。

## Frozen Retrieval And Packing

1. 使用 E3b workspace `snapshot-20260715T070507Z-e2ed57b0`；
2. document representation `cleaned_pdf_formula_text_v1`；
3. query representation `instructed_query_v1`；
4. embedding `Qwen/Qwen3-VL-Embedding-8B`，4096 dimensions；
5. Semantic@5、Hybrid@5 各执行一次；
6. ranking artifacts 保存后离线执行 E4b0 oracle 和冻结 E4b1；
7. 不执行 reranker，不改变 Hybrid 参数，不复用 canonical query vectors；
8. 新 query cache 独立保存，Hybrid 必须复用 Semantic 已生成的 20 vectors。

## Confirmation Gates

所有阈值在 query 生成前冻结：

1. `dataset_is_held_out`：10 spans 全部不在 canonical dataset，20 queries 无 canonical
   exact duplicate，provenance hashes 全部匹配。
2. `minimum_parent_hits`：Semantic@5、Hybrid@5 各至少 8 个 parent ranking hits；否则结果
   valid 但 evidence-retention confirmation 样本不足。
3. `oracle_evidence_reachable`：所有 parent hits 可评分，E4b0 evidence loss 为 0。
4. `fragment_hits_retained`：Semantic、Hybrid 的 E4b1 fragment evidence loss 均为 0。
5. `average_evidence_coverage`：两种 mode 各自 average reachable evidence coverage
   `>= 0.60`。
6. `minimum_evidence_coverage`：每个 parent hit 的 reachable evidence coverage
   `>= 0.25`。
7. `oracle_efficiency`：两种 mode 各自 average oracle efficiency `>= 0.80`。
8. `all_candidates_represented`：所有 case/mode 都按原顺序表示 Top-5 五个 candidates。
9. `all_fragments_traceable`：fragment text 与 workspace offset slice 完全一致。
10. `all_contexts_within_budget`：所有 rendered contexts nonempty 且 `<= 768` estimated
    tokens。
11. `complete_evidence_mapping`：mapping coverage 全部为 1.0。
12. `hybrid_context_tokens`：Hybrid average context-token ratio 相对 E0 724.5 `<= 1.10`。
13. `selector_gold_isolation`：E4b1 selector input allowlist 与 development 完全一致。

Gate 5-7 使用新 held-out cases，只比较 query selector、oracle 和 parent relevant chunks；
不根据 development cases 重新调整。Question type、difficulty、source、rank 和 page 分布全部
记录为 observational slices，不增加事后 gate。

## Decision Rule

- `valid: true` 要求 dataset、parent ranking、workspace、query cache、fragment provenance
  和 selector isolation 全部有效。
- `confirmed: true` 要求 `valid: true` 且 13 个 gates 全部通过。
- Gate 2 失败表示样本对 context retention 不足，不等同于 E4b 质量失败。
- 通过后 E4b1 才有资格进入 production runtime integration 和 answer-quality evaluation；
  仍不直接成为默认，也不直接进入 release baseline。
- 任一 quality gate 失败时保留完整 dataset/ranking/fragment artifacts，不删除困难 cases，
  后续 variant 必须使用新 ID 和新预注册协议。

## Commit Boundaries

1. 本协议与 frozen span selection；
2. dataset generation harness、tests 和 manifest；
3. generated held-out dataset、manual review 和 frozen hashes；
4. confirmation runner、tests 和 execution manifest；
5. ranking/fragment artifacts、结论与下一阶段 decision。

每个边界分别 commit 并 push。
