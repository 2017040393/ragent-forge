# E4b Held-out Confirmation Conclusion

- 日期：2026-07-16
- 状态：completed-not-confirmed
- Runner commit：`41cb038e5d5c313e8c58d90930361ca7da74b92d`
- Frozen implementation：`ccc930ae27740fca1da706c08487cc66cbb5493f`
- Dataset：[E4b held-out dataset review](2026-07-16-e4b-heldout-dataset-review.md)
- Artifacts：[E4b-heldout-41cb038](../benchmarks/results/fragment-packing/E4b-heldout-41cb038)
- Summary SHA-256：`236a889f201f65a2e4acfc072f44998a27141de86eabfc6081d30756ec421b0d`

## Decision

本次独立 held-out 运行 `valid: true`、`confirmed: false`。13 个预注册 gates 中 10 个
通过、3 个失败：`minimum_parent_hits`、`average_evidence_coverage` 和
`minimum_evidence_coverage`。因此 E4b1 不能进入 production runtime integration 或
answer-quality evaluation，也不能设为默认 context-selection policy。

## Result

| Metric | Semantic@5 | Hybrid@5 | Gate |
|---|---:|---:|---|
| Parent ranking hits | 7/20 | 12/20 | each >= 8 |
| Oracle evidence losses | 0 | 0 | 0 |
| E4b1 fragment evidence losses | 0 | 0 | 0 |
| Average reachable evidence coverage | 0.5091 | 0.5868 | each >= 0.60 |
| Minimum reachable evidence coverage | 0.0556 | 0.4843 | every hit >= 0.25 |
| Average oracle efficiency | 0.8080 | 0.9439 | each >= 0.80 |
| Average selector context tokens | 707.95 | 731.40 | Hybrid ratio <= 1.10 |

Hybrid context-token ratio 相对 E0 `724.5` 是 `1.0095`。两种 mode 的 Top-5 candidates
均按原顺序完整表示，fragment 全部可追踪，rendered context 全部不超过 768 estimated
tokens，evidence mapping coverage 全部为 1.0，selector allowlist 与 development 完全
一致。独立 query cache 在 Semantic 产生 `0 hits / 20 misses`，Hybrid 复用为
`20 hits / 0 misses`。

Semantic 只有 7 个 parent hits，低于预注册的 8 个最低样本量。按 decision rule，这一项
表示 Semantic 对 retention confirmation 的样本不足，不单独解释成 E4b1 质量失败。
Hybrid 有 12 个 parent hits，样本量足够，且 12 个 parent hits 全部保留；但 average
coverage `0.5868` 仍低于 `0.60`，所以 E4b1 在样本充分的 mode 上也没有通过质量 gate。

## Failure Diagnosis

`e4b-heldout-000019` 是 Semantic minimum coverage 的唯一失败 case。Semantic ranking
在 rank 3 命中一个映射到 evidence 的相邻 chunk；该 chunk 开头只保留了 evidence span
末尾关于 `F^n` 的内容，后半转入 `addition in F^n`。Oracle window 从 chunk 开头保留
36/36 reachable n-grams，query selector 被后半段重复的 `F^n` token 吸引，只保留
2/36，coverage 和 oracle efficiency 都是 `0.0556`。这不是候选长度偏置，而是同一
相关 chunk 内的 boundary/section disambiguation 失败。

作为运行后诊断而非 gate，Semantic/Hybrid 的平均 oracle coverage 分别为 `0.6794` 和
`0.6226`；Hybrid 有 5/12 个 parent hits 连 oracle 640-character window 都低于 `0.60`。
这说明失败由两个因素叠加：单 fragment 的固定最大长度使 coverage 上限接近阈值，query
selector 又在部分 case 上没有达到 oracle window。Hybrid selector 去掉最差 case 后平均
coverage 仍只有 `0.5961`，不能把失败归因于单一异常值。

## Next Decision

E4b1 保留为失败的 frozen candidate，不在当前 held-out 上调参。若继续 E4b2，需要新 ID
和新的预注册 development protocol，重点研究同一 candidate 的 boundary-aware 或
multi-window representation，同时继续表示全部 Top-5 candidates并保持 768-token 总预算。
当前 20 held-out cases 已被观察，只能作为后续 development/diagnosis 数据；任何 E4b2
独立 confirmation 都必须重新生成未观察 spans/cases。

复现本结果：

```powershell
uv run --extra dev python -m benchmarks.fragment_packing_confirmation `
  --workspace .ragent/baselines/E3b-pdf-formula-3e5758c `
  --output-dir .ragent/eval/fragment-packing/E4b-heldout-41cb038-replay
```
