# E4b Fragment Packing Development Conclusion

- 日期：2026-07-16
- 状态：development-passed-not-confirmed
- Variant：`E4b1-ranked-query-fragments`
- Evaluation commit：`ccc930ae27740fca1da706c08487cc66cbb5493f`
- Specification：[E4b multi-candidate fragment packing](2026-07-16-e4b-multi-candidate-fragment-packing.md)
- Artifacts：[E4b1-50-case-ccc930a](../benchmarks/results/fragment-packing/E4b1-50-case-ccc930a)

## Result

本轮在 clean commit 上纯离线重放 E4a 保存的 50-case Semantic@5、Hybrid@5 ranking，
没有调用 embedding provider，也没有改变 candidate IDs、scores 或 order。Parent、dataset、
workspace、chunk/index fingerprints 和 selector input allowlist 全部验证通过。

最终 `valid: true`、九个 development gates 全部通过、`development_passed: true`；由于这
50 cases 已用于设计 E4b，报告固定为 `confirmation_claim_allowed: false`。

| Mode | Parent ranking hits | Scorable | Oracle losses | E4b1 losses | Avg context tokens | Max tokens |
|---|---:|---:|---:|---:|---:|---:|
| Semantic@5 | 31 | 31 | 0 | 0 | 703.26 | 767 |
| Hybrid@5 | 35 | 35 | 0 | 0 | 742.80 | 767 |

Hybrid context-token ratio 相对 E0 的 724.5 为 `1.0253`，低于 `1.10` gate。100 个
mode/case combinations 全部表示原 Top-5 五个 candidates，顺序不变；所有 fragment
offset 与 workspace text slice 完全相等，rendered header、正文和 separators 均计入
3072-character 预算，parent mapping coverage 保持 1.0。

## Evidence Preservation

E4b0 oracle 在 66 个 mode-specific parent hits 上全部找到 evidence，unscorable hit 为 0。
E4b1 query-only selector 也在全部 66 个 hits 上找到至少一个 gold three-token n-gram，
没有沿用“chunk ID 出现即算命中”的 E4a 口径。

| Mode | Avg reachable evidence coverage | Minimum coverage | Avg oracle efficiency |
|---|---:|---:|---:|
| Semantic@5 | 0.7026 | 0.3585 | 0.8833 |
| Hybrid@5 | 0.6802 | 0.3208 | 0.8817 |

四个 E4a Hybrid selection losses 均恢复 fragment evidence：

| Case | Relevant rank | Coverage | Oracle efficiency | Rendered tokens |
|---|---:|---:|---:|---:|
| `v0-2-baseline-000009` | 4 | 0.5370 | 0.9355 | 761 |
| `v0-2-baseline-000021` | 5 | 1.0000 | 1.0000 | 737 |
| `v0-2-baseline-000039` | 5 | 0.3208 | 0.3269 | 760 |
| `v0-2-baseline-000042` | 5 | 0.6316 | 1.0000 | 763 |

其中 `000039` 同时是当前最弱的 Hybrid sample。Selector 确实选到了 quotient-space
evidence，但只达到 oracle 可覆盖 n-grams 的约三分之一。该结果不会事后改变已经冻结的
development gates；它必须作为 held-out 阶段的显式风险和 coverage gate 依据。

## Interpretation

E4b0 证明 768-token、五候选、连续 fragment contract 在当前 parent hits 上可行。E4b1
进一步证明，仅使用 query 与 chunk text 的确定性窗口评分能够解决 E4a 的 whole-chunk
packing 失败，而不需要重新 embedding、重排或 gold-assisted runtime。

代价是 context 使用接近上限：Hybrid average 742.8 tokens，比 E4a 的 692.34 更高，但
保住了 E4a 丢失的 4 个 hits；相对 E0 只增加约 2.53%，仍在预注册 gate 内。当前结果只
评价 retrieval evidence preservation，不评价 fragment 输入 LLM 后的 answer quality。

## Decision

冻结 `ranked_query_fragment_budget_v1` 和 commit `ccc930a`，进入新的 held-out
confirmation 设计。E4b1 暂不接入默认 runtime，不进入 release baseline，也不根据当前
50-case 的 coverage 分布继续调参。

Held-out confirmation 必须使用未参与 E4b 设计的新 evidence spans/cases，并预先冻结：

- parent E3b+E2 ranking 生成流程；
- zero fragment-evidence loss；
- average reachable evidence coverage 与 oracle efficiency gates；
- 768-token、Top-5 representation、traceability 和 selector-gold-isolation gates；
- 独立 answer-quality evaluation 仍作为后续阶段，而不是从本轮 retrieval metrics 推断。

## Reproduction

```powershell
uv run --extra dev python -m benchmarks.fragment_packing_development `
  --manifest benchmarks/fragment_packing_development_manifest_e4b.json `
  --workspace .ragent/baselines/E3b-pdf-formula-3e5758c `
  --output-dir .ragent/eval/fragment-packing/E4b1-50-case-ccc930a-replay
```
