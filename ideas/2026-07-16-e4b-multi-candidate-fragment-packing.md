# E4b Multi-candidate Fragment Packing

- 日期：2026-07-16
- 状态：development-passed-not-confirmed
- 阶段：pre-v0.3 context-selection development
- Parent：[E4a 50-case conclusion](2026-07-15-e4a-50-case-direction-confirmation-conclusion.md)
- Parent artifacts：[E4a-50-case-8299040](../benchmarks/results/direction-confirmations/E4a-50-case-8299040)
- Result：[E4b development conclusion](2026-07-16-e4b-fragment-packing-development-conclusion.md)
- Artifacts：[E4b1-50-case-ccc930a](../benchmarks/results/fragment-packing/E4b1-50-case-ccc930a)

## 目的与证据边界

E4a 已证明 `ranked_prefix_token_budget_v1` 的 whole-chunk strict-prefix 形态无法同时
满足 Hybrid hit retention 和 context-token gate。E4b 保持冻结的 E3b+E2 Top-5 ranking，
只把 context unit 从完整 chunk 改为可追踪的连续 fragment，使 rank 4/5 候选也能在
768 estimated-token 预算内获得上下文份额。

当前 50-case dataset、四个 E4a Hybrid loss cases 和 budget frontier 已经被人工查看。
因此本轮重放只能作为 development evidence，不能再次命名为独立 confirmation。E4b
参数冻结后，必须用新的 held-out cases 才能形成 confirmation claim。

E4b 不修改 document/query representation、chunk boundaries、index、embedding vectors、
BM25/Hybrid 参数、candidate score、candidate ID 或 candidate rank，也不调用 embedding
provider。

## 两层实验

### E4b0 Oracle Feasibility

E4b0 使用与 E4b1 相同的候选窗口、预算分配和渲染格式，但允许 evaluator 使用 gold
evidence span 为每个 chunk 选择覆盖量最高的窗口。它只回答：在相同 fragment contract
和 768-token 预算下，理论上是否存在能保留 parent evidence 的 packing。

E4b0 不是生产 selector，不可 promotion，不得进入 runtime。它的结果必须标记为
`oracle_only: true`。

### E4b1 Query-guided Selector

E4b1 的 selector 输入只能包含原始 query、冻结 ranking 和 workspace chunk records。
它不得接收 evidence span、reference answer、mapped expected chunk IDs、relevant ranks 或
failure labels。Gold data 只在 selector 返回之后由 evaluator 使用。

Variant ID：`E4b1-ranked-query-fragments`

Selection policy：`ranked_query_fragment_budget_v1`

## Frozen Fragment Contract

固定参数：

| Parameter | Value |
|---|---:|
| Candidate depth | 5 |
| Max estimated context tokens | 768 |
| Characters per estimated token | 4 |
| Max rendered context characters | 3072 |
| Maximum fragment characters | 640 |
| Internal long-unit stride | 112 characters |
| Evidence n-gram size | 3 normalized tokens |

每个 Top-5 candidate 必须产生恰好一个连续 fragment。Fragment 保存 `chunk_id`、原始
rank、portable source label、page label、chunk-relative `start_char/end_char`、原始文本、
截断标记和 score components。正文必须严格等于原 chunk 对应字符区间，不允许生成、
改写或补全文本。

渲染 header 固定为：

```text
[rank=<rank> source=<portable filename> page=<page label>]
```

五个 headers、五个 fragment texts 和四个双换行 separator 全部计入 3072-character
预算。先从总预算扣除 headers 和 separators，再对五个 candidates 做 deterministic
water-filling：当前仍可增长的 candidates 等分剩余 content budget，短 chunk 的未用份额
继续等分给其他 candidates；单个 fragment 不超过 640 characters。余数按 rank 从小到大
分配。不得因 query score 改变不同 candidates 的预算或顺序。

## Candidate Windows And Scoring

候选窗口必须覆盖完整 chunk，包括首尾窗口。Window boundaries 优先来自换行、段落和
句末边界；超过 fragment allocation 的长 unit 使用 112-character stride 生成内部窗口。
每个窗口都是一个连续 `[start_char, end_char)` 区间，长度不超过该 candidate allocation。

E4b1 只在同一个 chunk 的窗口之间选择。冻结 score tuple 按以下顺序最大化：

1. 去除固定英文 stopwords 后的 unique query-token coverage；
2. query-token occurrence count；
3. normalized query bigram coverage；
4. heading/formula signal token coverage；
5. 更长的窗口；
6. 更早的 start offset。

所有 tokenization 使用 Unicode NFKC、casefold 和 `\w+`。Score 相同按上述 tuple 的后续
字段决定，不允许随机 tie-break。Query score 只选择 candidate 内部窗口，不重排 Top-5。

## Fragment-level Evidence Evaluation

Canonical 50 cases 每条都有一个 evidence span text，中位长度约 1197 characters。E4b
不以“相关 chunk ID 被列出”作为 fragment hit；evaluator 对 gold text、parent relevant
chunks、oracle fragment 和 E4b1 fragment 使用同一 NFKC/casefold token normalization，
并计算三 token n-gram 集合：

- `reachable_evidence_ngrams`：gold n-grams 与 parent relevant Top-5 chunk texts 的交集；
- `selected_evidence_ngrams`：gold n-grams 与实际相关 fragments 的交集；
- `oracle_evidence_ngrams`：gold n-grams 与同 allocation 下 oracle fragments 的交集；
- `reachable_evidence_coverage`：selected / reachable；
- `oracle_efficiency`：selected / oracle。

当 parent ranking hit 且 `reachable_evidence_ngrams > 0` 时，只有
`selected_evidence_ngrams > 0` 才算 fragment evidence retained。没有 reachable n-gram
的 case 标记为 unscorable，不得静默算作通过。E4b0 必须同时报告窗口候选生成是否让
oracle 找到 evidence。

## Development Gates

本轮 50-case replay 的 gates 在运行前固定为：

1. `oracle_evidence_reachable`：所有可评分的 parent ranking hits 在 E4b0 中都保留
   evidence，且 unscorable hit 数为 0。
2. `semantic_fragment_hits_retained`：E4b1 相对 31 个 Semantic ranking hits 的 fragment
   evidence loss 为 0。
3. `hybrid_fragment_hits_retained`：E4b1 相对 35 个 Hybrid ranking hits 的 fragment
   evidence loss 为 0。
4. `all_candidates_represented`：100 个 mode/case combinations 都包含原 Top-5 五个
   candidate IDs，顺序不变。
5. `all_fragments_traceable`：所有 fragment 区间有效，文本与 workspace slice 完全相等。
6. `all_contexts_within_budget`：所有 rendered contexts nonempty 且不超过 3072 chars / 768
   estimated tokens。
7. `complete_evidence_mapping`：parent mapping coverage 全部为 1.0。
8. `hybrid_context_tokens`：Hybrid average estimated tokens 相对 E0 parent 724.5 的 ratio
   `<= 1.10`。
9. `selector_gold_isolation`：E4b1 artifact 记录 selector input allowlist，且不含任何 gold
   evidence 或 expected-result 字段。

同时记录 average/minimum `reachable_evidence_coverage`、`oracle_efficiency`、fragment
length distribution、source/rank distribution 和四个 E4a loss cases 的逐项变化，但不增加
事后 gate。

## Decision Rule

- `valid: true` 要求 parent hashes、dataset/workspace fingerprints、fragment provenance、
  selector isolation 和全部 artifact contracts 有效。
- `development_passed: true` 要求 `valid: true` 且九个 development gates 全部通过。
- 通过只表示 E4b1 值得冻结并进入新 held-out confirmation；不得设为默认，不得进入
  release baseline，也不得形成 answer-quality claim。
- Oracle 通过而 E4b1 失败，说明 query-guided window scorer 不足；两者都失败则说明当前
  fragment contract、evidence mapping 或 768-token budget 本身不可行。

## Expected Artifacts

- resolved manifest 和 parent ranking hashes；
- workspace chunk/index fingerprints；
- E4b0/E4b1 Semantic@5、Hybrid@5 fragment artifacts；
- 每个 fragment 的 offset、rendered text、score tuple 和 provenance；
- fragment-level evidence metrics、九个 gates、summary 和 selector input allowlist；
- 独立 development conclusion 和后续 held-out specification。

## Recorded Result

本协议已在 clean commit `ccc930ae27740fca1da706c08487cc66cbb5493f` 上执行。
结果 `valid: true`、九个 development gates 全部通过、`development_passed: true`，但
`confirmation_claim_allowed: false`。Semantic 31/31、Hybrid 35/35 parent ranking hits
均保留 fragment evidence；Hybrid average context 为 742.8 tokens，E0 ratio 为 1.0253。
完整 coverage、oracle gap 和限制见结论文档。
