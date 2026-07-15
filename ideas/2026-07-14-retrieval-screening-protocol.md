# Retrieval Representation Screening Protocol

- 日期：2026-07-14
- 状态：implemented
- 相关版本：pre-v0.3 representation experiments
- Workflow commit：`8a0544d`
- Screening manifest：
  [`retrieval_screen_manifest.json`](../benchmarks/retrieval_screen_manifest.json)
- E0 summary：
  [`summary.json`](../benchmarks/results/screens/E0-raw-text-8a0544d/summary.json)
- Parent baseline：
  [post-architecture baseline](2026-07-14-pre-v0-3-post-architecture-baseline.md)

## 目的与边界

快速筛选用于在完整 50-case 和正式 36-trial matrix 之前淘汰没有方向性收益的
retrieval representation variants。它是 decision filter，不是 release benchmark，
也不能产生 latency claim。

筛选只减少 query 数量，不减少 corpus。每个 variant 仍使用完整的 4 documents、
1744 chunks 和 4096-dimensional index，从而保留长 PDF 内的 cross-section
distractors。缩小 corpus 会掩盖当前主要的 `wrong_section` failure mode。

正式 baseline 的 `BaselineWorkloadSpec` 仍要求至少三轮。Screening 使用独立 typed
contract，固定为一轮，不能通过 screening 参数降低正式 baseline 的重复次数。

## 固定 Diagnostic Slice

Manifest 从 canonical 50-case dataset 中按 ID 选择 16 条。14 条参与 gates，2 条
boundary canaries 只观察 dense ranking 波动。

| Group | Case IDs | Parent behavior | Purpose |
|---|---|---|---|
| Stable control | `000001`、`000019`、`000013`、`000022` | Semantic@5 3/3 hit | 防止基础能力回退 |
| Semantic opportunity | `000016`、`000041` | Semantic@5 3/3 hit，BM25 miss | 保留 semantic 独立价值 |
| Wrong-section challenge | `000003`、`000031`、`000017`、`000046` | BM25 hit，Semantic@5 0/3 | 直接检验 structured representation |
| Hard miss | `000005`、`000006`、`000036`、`000040` | 主要 modes 均 miss | 观察真正的新能力 |
| Boundary canary | `000014`、`000033` | Hybrid@5 分别 2/3、1/3 hit | 观察波动，不参与晋级 |

完整 case IDs 使用 `v0-2-baseline-` 前缀。Source distribution 为 2 条 Markdown、
6 条 probability PDF、8 条 linear algebra PDF，并覆盖 definition、formula、
reasoning、comparison 和 cross-section distractors。

## Workload 与 Cache Boundary

每个 variant 独立执行：

```text
semantic@5
semantic@20
hybrid@5
hybrid@20
```

Requested limits 5 和 20 必须独立执行，因为 candidate depth 与 hybrid fusion 会随
requested limit 改变，不能从 top-20 截取 top-5。

四组配置共享一个 snapshot-keyed prepared-state cache，因此 chunks 和 vector index
只加载一次。16 个 raw queries 只向 provider 获取一次 embeddings；随后三个配置
复用冻结的 query vectors。相同 query representation 的 candidate 通过
`--query-cache-source` 读取 E0 cache，并写入自己的输出目录，不修改 source cache。

因为 cache reuse 改变了 timing boundary，screening latency 只用于发现异常，不能与
正式 cold/warm baseline 比较。真实 provider 和 release latency 仍由正式 matrix
测量。

## Promotion Gates

Candidate 必须同时满足：

1. Stable controls 与 semantic opportunities 的 Semantic@5 loss 为 0。
2. 不产生新的 `missed_source` result。
3. Challenge/hard cases 至少新增 1 个 Semantic@5 hit，或至少新增 2 个
   Semantic@20 hit。
4. 14 个 gated cases 的 Hybrid@5 net hit delta 不低于 0。
5. Gated Hybrid@5 average selected tokens 不超过 parent 的 1.10 倍。
6. 所有 selected results 的 evidence mapping coverage 为 1.0。

Boundary canaries 的 hit、rank 和 fingerprint 会保留在 artifacts 中，但不参与这些
gates。通过 screening 只允许进入 50-case direction confirmation，不等于达到 v0.3
release gates。

## E0 Frozen Result

E0 使用 raw chunk text、raw query 和正式 baseline workspace：

| Item | Value |
|---|---|
| Git commit | `8a0544d` |
| Workspace build commit | `ca029f9` |
| Workspace snapshot | `snapshot-20260714T024249Z-60c124c1` |
| Variant | `E0-raw-text`，role `baseline` |
| Query cache | 16 entries，4096 dimensions |
| Cache behavior | 16 misses，48 hits |
| Artifact count / size | 7 files / 3,136,764 bytes |
| Structural result | `valid: true` |
| Promotion | not applicable |

Quality comparison：

| Configuration | Parent 16-case Hit | E0 Hit | Parent 14-gated Hit | E0 gated Hit |
|---|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.4375 | 0.4286 | 0.4286 |
| Semantic@20 | 0.5625 | 0.5625 | 0.5714 | 0.5714 |
| Hybrid@5 | 0.6250 | 0.5625 | 0.6429 | 0.6429 |
| Hybrid@20 | 0.7500 | 0.7500 | 0.7143 | 0.7143 |

Hybrid@5 的全切片差异只来自两个 boundary canaries：E0 本轮均未命中，而 parent
三轮分别是 2/3 和 1/3。排除 canaries 后，14 个 gated cases 的 Hybrid@5 Hit
完全一致，说明 gate 没有把已知 dense boundary variability 误判为基础回退。

E0 的 `semantic_challenge_gain` 为 fail，因为 raw-text baseline 相对自身没有新增
challenge hit。这是 reference variant 的预期结果。E0 role 为 `baseline`，所以
`promotion_applicable: false`、`promoted: null`；其余结构与回退 gates 均通过。

## E1 Candidate Result

E1 使用 `structured_document_text_v1` 生成 document-side embedding input，保留
E0 的 raw query、完整 1744-chunk corpus 和 16-case screen。结果 artifacts：
[`E1-structured-document-c5e71b2`](../benchmarks/results/screens/E1-structured-document-c5e71b2)，
candidate manifest：
[`retrieval_screen_manifest_e1.json`](../benchmarks/retrieval_screen_manifest_e1.json)。

| Item | Value |
|---|---|
| Evaluation commit | `c5e71b2` |
| Workspace build commit | `a43695b` |
| Workspace snapshot | `snapshot-20260715T033549Z-a1f9077f` |
| Chunk content fingerprint | unchanged: `a99c38a3...3278a` |
| Index input fingerprint | `46d8a7b0...c24bff` |
| Query cache | 16 entries, 64 hits, 0 misses |
| Structural result | `valid: true` |
| Promotion | rejected |

Quality comparison：

| Configuration | E0 full hit | E1 full hit | E0 gated hit | E1 gated hit |
|---|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.3750 | 0.4286 | 0.3571 |
| Semantic@20 | 0.5625 | 0.5000 | 0.5714 | 0.5000 |
| Hybrid@5 | 0.5625 | 0.6250 | 0.6429 | 0.5714 |
| Hybrid@20 | 0.7500 | 0.6875 | 0.7143 | 0.6429 |

E1 通过 `no_new_missed_source`、`semantic_challenge_gain` 和
`complete_evidence_mapping`，其中 challenge cases 新增 2 个 Semantic@5 hit。
但它有 3 个 stable Semantic@5 losses（`000013`、`000016`、`000022`），gated
Hybrid@5 相对 parent 减少 1 个 hit，selected context tokens ratio 为 `1.3707`
（上限 `1.10`）。因此 E1 不能进入 50-case direction confirmation，也不能作为
v0.3 的候选 representation 晋级。

## E2 Candidate Result

E2 复用 E1 的 `structured_document_text_v1` index，只把 query-side input 改为
`instructed_query_v1`。结果 artifacts：
[`E2-instructed-query-6c453bb`](../benchmarks/results/screens/E2-instructed-query-6c453bb)，
candidate manifest：
[`retrieval_screen_manifest_e2.json`](../benchmarks/retrieval_screen_manifest_e2.json)。

| Item | Value |
|---|---|
| Evaluation commit | `6c453bb` |
| Workspace build commit | `a43695b` |
| Workspace snapshot | `snapshot-20260715T033549Z-a1f9077f` |
| Query representation | `instructed_query_v1` |
| Query cache | 16 entries, 16 misses, 48 hits |
| Structural result | `valid: true` |
| Promotion | rejected |

Quality comparison：

| Configuration | E0 full hit | E2 full hit | E0 gated hit | E2 gated hit |
|---|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.5000 | 0.4286 | 0.5000 |
| Semantic@20 | 0.5625 | 0.6250 | 0.5714 | 0.5714 |
| Hybrid@5 | 0.5625 | 0.5625 | 0.6429 | 0.5714 |
| Hybrid@20 | 0.7500 | 0.6875 | 0.7143 | 0.6429 |

Query instruction 相对 E1 改善了 semantic ranking：新增 3 个 Semantic@5 challenge
hits 和 2 个 Semantic@20 challenge hits，并恢复了 E1 丢失的 `000013`。这说明
document/query embedding input 不对称确实是 E1 下降的一部分原因。

但 E2 仍丢失 `000016` 和 `000022` 两个 stable Semantic@5 cases；gated Hybrid@5
相对 parent 减少 1 个 hit，selected context tokens ratio 为 `1.3378`。因此 E2 不能
进入 50-case direction confirmation，也不能作为当前 v0.3 representation 晋级。

## E3a Candidate Result

E3a 保留 E2 的 `instructed_query_v1`，只把 PDF document representation 改为
`cleaned_pdf_section_text_v1`，加入确定性 PDF layout cleaning 和跨 chunk section
cue propagation。结果 artifacts：
[`E3a-pdf-section-2930f68`](../benchmarks/results/screens/E3a-pdf-section-2930f68)，
candidate manifest：
[`retrieval_screen_manifest_e3a.json`](../benchmarks/retrieval_screen_manifest_e3a.json)。

| Item | Value |
|---|---|
| Evaluation commit | `2930f68` |
| Workspace build commit | `e68c6ce` |
| Workspace snapshot | `snapshot-20260715T055327Z-4107a518` |
| Document representation | `cleaned_pdf_section_text_v1` |
| Query representation | `instructed_query_v1` |
| Query cache | 16 entries, 64 hits, 0 misses |
| Structural result | `valid: true` |
| Promotion | rejected |

Quality comparison：

| Configuration | E0 full hit | E3a full hit | E0 gated hit | E3a gated hit |
|---|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.6250 | 0.4286 | 0.6429 |
| Semantic@20 | 0.5625 | 0.8125 | 0.5714 | 0.8571 |
| Hybrid@5 | 0.5625 | 0.6875 | 0.6429 | 0.7143 |
| Hybrid@20 | 0.7500 | 0.8125 | 0.7143 | 0.7857 |

E3a 通过 `no_new_missed_source`、`semantic_challenge_gain`、
`hybrid_top5_net_nonnegative` 和 `complete_evidence_mapping`。它在 challenge/hard
cases 新增 4 个 Semantic@5 hit，Hybrid@5 gated hit 相对 parent 增加 1 个，说明
PDF section cues 是当前最有方向性的改进。

但 E3a 仍有 `000016` 一个 stable Semantic@5 loss，且 selected context tokens ratio
为 `1.4185`，超过 `1.10`。因此 E3a 不能晋级；E3b 继续完整复用 E3a，只增加
高置信度 formula evidence。

## E3b Candidate Result

E3b 完整复用 E3a 的 PDF cleaning、section cues 和 E2 的
`instructed_query_v1`，只增加从当前 chunk `possible_formula_lines` 生成的确定性
`Formula evidence`。结果 artifacts：
[`E3b-pdf-formula-d21c91b`](../benchmarks/results/screens/E3b-pdf-formula-d21c91b)，
candidate manifest：
[`retrieval_screen_manifest_e3b.json`](../benchmarks/retrieval_screen_manifest_e3b.json)。

| Item | Value |
|---|---|
| Evaluation commit | `d21c91b` |
| Workspace build commit | `3e5758c` |
| Workspace snapshot | `snapshot-20260715T070507Z-e2ed57b0` |
| Document representation | `cleaned_pdf_formula_text_v1` |
| Query representation | `instructed_query_v1` |
| Chunk content fingerprint | unchanged: `a99c38a3...3278a` |
| Index input fingerprint | `48b5f9be...ab9e4` |
| Query cache | 16 entries, 64 hits, 0 misses |
| Structural result | `valid: true` |
| Promotion | rejected |

Quality comparison 以 manifest 冻结的三轮 parent baseline mean 为 E0 reference；
E3a 仅作方向对照：

| Configuration | E0 parent hit | E3a hit | E3b hit | E0 parent gated | E3b gated |
|---|---:|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.6250 | 0.6875 | 0.4286 | 0.7143 |
| Semantic@20 | 0.5625 | 0.8125 | 0.8125 | 0.5714 | 0.7857 |
| Hybrid@5 | 0.6250 | 0.6875 | 0.6250 | 0.6429 | 0.6429 |
| Hybrid@20 | 0.7500 | 0.8125 | 0.8750 | 0.7143 | 0.8571 |

E3b 保留全部 stable Semantic@5 hits，并在 challenge/hard cases 相对 E0 parent
新增 4 个 Semantic@5 hit 和 3 个 Semantic@20 hit；没有新增 `missed_source`，
Hybrid@5 gated hit 与 parent 持平，evidence mapping coverage 保持完整。相对 E3a，
它恢复了 `000016` 并提高 Hybrid@20，但不再保留 `000036` 的 Hybrid@5 增益。

唯一失败 gate 是 `hybrid_top5_context_tokens`：ratio 从 E3a 的 `1.4185` 改善到
`1.3690`，仍高于 `1.10`。因此 E3b 不进入 50-case direction confirmation。

## E4a Context Selection Result

E4a 不重新生成 E3b vectors，而是重放 E3b 已保存的 Semantic@5 和 Hybrid@5
ranking，只应用 `ranked_prefix_token_budget_v1`：按原 rank 取完整 chunk 前缀，
不截断、不跳过、不回填，最大 `768` estimated tokens（`3072` characters）。
结果 artifacts：
[`E4a-ranked-prefix-token-budget-ed8d609`](../benchmarks/results/context-screens/E4a-ranked-prefix-token-budget-ed8d609)，
manifest：
[`context_selection_screen_manifest_e4a.json`](../benchmarks/context_selection_screen_manifest_e4a.json)。

| Item | Value |
|---|---|
| Evaluation commit | `ed8d609` |
| Parent variant | `E3b-pdf-formula` |
| Selection policy | `ranked_prefix_token_budget_v1` |
| Max context | `768` estimated tokens / `3072` characters |
| Semantic@5 parent hits retained | `11/11` |
| Hybrid@5 parent hits retained | `10/10` |
| Hybrid@5 gated average | `702.0714` tokens |
| Hybrid@5 gated token ratio | `0.9683` |
| Structural result | `valid: true` |
| Promotion | `promoted: true` |

E4a 通过全部六个 context-selection gates：没有丢失 E3b 的 Semantic/Hybrid Top-5
相关命中，所有 case 至少保留一个完整 chunk，所有上下文在预算内，且 selected IDs
始终是 parent ranking 的严格前缀。它修复了 E3b 唯一失败的 context-token gate，
但只证明固定 16-case screen 上的 context packing 行为，不等于完整 v0.3 release 结论。

## Reproduction

```powershell
uv run --extra dev python -m benchmarks.retrieval_screen `
  --workspace .ragent/baselines/pre-v0.3-ca029f9 `
  --output-dir benchmarks/results/screens/E0-raw-text-8a0544d
```

Runner 会验证 parent summary hash、dataset/corpus hashes、workspace generation
layout、chunk/index counts、embedding configuration，以及跨机器稳定的 chunk-content
和 index-input fingerprints。`--resume` 还会验证 manifest、Git commit、workspace
snapshot、variant、mode、limit、case set 和已有 run artifact。

## 下一步

E3a/E3b/E4a screen 和后续 50-case direction confirmation 均已完成。完整集上 E3b+E2
ranking 达到 Semantic@5 `0.62`、Hybrid@5 `0.70`，但 E4a 768-token prefix 丢失 4 个
Hybrid ranking hits，因此结果为 `valid: true`、`confirmed: false`。详见
[50-case conclusion](2026-07-15-e4a-50-case-direction-confirmation-conclusion.md)。

下一步不继续微调 whole-chunk fixed budget；离线 frontier 已证明零丢失至少需要 1207
tokens，此时 Hybrid token ratio 为 `1.3904`。应冻结本轮 ranking artifacts，设计带
evidence-preservation 检查的 E4b 多候选片段 packing，再离线重放；当前组合不设为默认，
也不进入正式 release baseline。
