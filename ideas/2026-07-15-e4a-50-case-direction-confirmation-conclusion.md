# E4a 50-case Direction Confirmation Conclusion

- 日期：2026-07-15
- 状态：completed-not-confirmed
- Candidate：`E3b-pdf-formula + E2-instructed-query + E4a-ranked-prefix`
- Harness commit：`8299040a04ae32f5d5d632713d45f14d6759e9c8`
- Specification：[E4a 50-case direction confirmation](2026-07-15-e4a-50-case-direction-confirmation.md)
- Artifacts：[E4a-50-case-8299040](../benchmarks/results/direction-confirmations/E4a-50-case-8299040)

## Result

本轮输入与 provenance 校验全部通过，`valid: true`；七个冻结 gates 中六个通过，
`hybrid_context_hits_retained` 失败，因此最终为 `confirmed: false`。

| Mode | E0 Top-5 hit rate | Candidate ranking | Delta | E4a selected hits | Average selected tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic | 0.26 | 0.62 (31/50) | +0.36 | 31/31 | 671.38 |
| Hybrid | 0.64 | 0.70 (35/50) | +0.06 | 31/35 | 692.34 |

Query cache 按协议从 16-entry E2 seed 扩充到 50 entries：Semantic 为 16 hits、
34 misses，Hybrid 为 50 hits、0 misses。最终 16 个 seed vectors 保持不变，ranking、
query cache 与 resume checkpoint hashes 全部闭合。

Semantic parent-consensus transitions 为 13 retained、18 gained、0 lost、19 unchanged
miss；Hybrid 为 28 retained、7 gained、4 lost、11 unchanged miss。两种 mode 都没有新增
`missed_source`。因此 E3b+E2 ranking 方向得到完整数据集支持，但这不等于 E4a context
selection 也通过。

## Failed Retention Gate

E4a 的 768-token whole-chunk strict prefix 丢失了 4 个 Hybrid ranking hits：

| Case | Relevant rank | Selected prefix | Tokens needed through relevant chunk |
| --- | ---: | ---: | ---: |
| `v0-2-baseline-000009` | 4 | 1-3 | 846 |
| `v0-2-baseline-000021` | 5 | 1-3 | 922 |
| `v0-2-baseline-000039` | 5 | 1-3 | 1079 |
| `v0-2-baseline-000042` | 5 | 1-3 | 1207 |

这些 case 都不是 mapping 或预算违规：mapping coverage 为 1.0，selected IDs 始终是
ranking 的严格前缀，且实际选中内容没有超过 768 tokens。失败原因是较长的高排名 chunks
先占满预算，使第 4/5 名相关 chunk 无法进入 context。

Hybrid selected-count distribution 为 35 个 case 选择 3 chunks、14 个选择 4 chunks、
1 个选择 5 chunks。E4a 把 Hybrid 平均 context-token ratio 降到 `0.9556`，但同时把
35 个 ranking hits 裁成 31 个，抵消了 ranking 相对 E0 的净增益。

## Budget Frontier

对冻结 Hybrid ranking artifact 做了 `768..1400` 的纯离线整数预算扫描。完整结果保存在
[`budget-frontier.json`](../benchmarks/results/direction-confirmations/E4a-50-case-8299040/budget-frontier.json)。

- 保持 token ratio `<= 1.10` 的最大预算是 921 tokens，此时仍丢失 3 个 hits。
- 零 hit loss 的最小预算是 1207 tokens，此时 ratio 为 `1.3904`。
- 因此 whole-chunk strict-prefix 策略不存在同时满足 retention 和 token gates 的固定预算；
  继续微调 768 不会解决问题。

## Decision

不把 E4a 设为默认，也不进入正式多轮 release baseline。保留 E3b+E2 ranking 作为下一步
context-selection 研究的冻结 parent，停止继续做 whole-chunk fixed-budget sweep。

下一候选应改为 E4b：在不改变 Top-5 顺序的前提下，为多个候选分配受限片段或 evidence
windows，使后排候选仍能进入 context。E4b 必须增加 evidence-preservation 检查，不能只用
“selected chunk ID 存在”来把截断文本误判为命中。它可以完全离线重放本轮 50-case ranking
artifacts，设计与初筛阶段不需要再次调用 embedding provider。

该 development replay 已完成并通过九个预注册 gates，见
[E4b development conclusion](2026-07-16-e4b-fragment-packing-development-conclusion.md)。
因为本页的 50 cases 已被用于 E4b 设计，下一次 confirmation 必须换用新 held-out cases。

## Reproduction

从 harness commit 的干净工作区运行：

```powershell
uv run --extra dev python -m benchmarks.direction_confirmation `
  --manifest benchmarks/direction_confirmation_manifest_e4a.json `
  --workspace .ragent/baselines/E3b-pdf-formula-3e5758c `
  --output-dir .ragent/eval/direction-confirmations/E4a-50-case-8299040-replay
```
