# E3b Retrieval Screening Conclusion

- 日期：2026-07-15
- Variant：`E3b-pdf-formula`
- 状态：rejected
- Protocol：[Retrieval Representation Screening Protocol](2026-07-14-retrieval-screening-protocol.md)
- Specification：[E3 PDF representations](2026-07-15-e3-pdf-embedding-representations.md)
- Artifacts：[E3b screen result](../benchmarks/results/screens/E3b-pdf-formula-d21c91b)

## 实验边界

E3b 完整复用 E3a 的 PDF cleaning 和 section propagation，并继续使用 E2 的
`instructed_query_v1`。唯一 document-side 增量是
`cleaned_pdf_formula_text_v1`：从当前 chunk 的 `possible_formula_lines` 中筛选高置信度
公式，做 NFKC 和有限数学符号归一化，按原顺序去重后最多加入 3 条
`Formula evidence`。Markdown representation、`chunk.text`、chunk boundaries、BM25
input、模型、corpus、16-case slice、Hybrid 参数和 promotion gates 均未改变。

E3b workspace 使用独立 generation，包含 4 documents、1744 chunks 和 1744 条
4096-dimensional vectors。Chunk content fingerprint 保持
`a99c38a3...3278a`，index-input fingerprint 为 `48b5f9be...ab9e4`。E2 instructed
query cache 被完整复用，本轮为 64 hits、0 misses。

## 结果

下表的 E0 是 manifest 冻结的三轮 parent baseline mean；E3a 只用于观察 E3b 的
增量方向：

| Configuration | E0 parent hit | E3a hit | E3b hit | E0 parent gated | E3b gated |
|---|---:|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.6250 | 0.6875 | 0.4286 | 0.7143 |
| Semantic@20 | 0.5625 | 0.8125 | 0.8125 | 0.5714 | 0.7857 |
| Hybrid@5 | 0.6250 | 0.6875 | 0.6250 | 0.6429 | 0.6429 |
| Hybrid@20 | 0.7500 | 0.8125 | 0.8750 | 0.7143 | 0.8571 |

通过的 gates：

- stable Semantic@5 loss 为 0；
- 没有新增 `missed_source`；
- challenge/hard cases 新增 4 个 Semantic@5 hit 和 3 个 Semantic@20 hit；
- gated Hybrid@5 与 parent 持平；
- evidence mapping coverage 保持 1.0。

失败的 gate：

- selected context tokens ratio 为 `1.3690`，超过 `1.10`。

## Interpretation

E3b 修复了 E3a 对 stable case `000016` 的损失，并把 Hybrid@20 从 `0.8125`
提高到 `0.8750`。这说明显式 formula representation 对数学 PDF 的章节内区分有补充
价值，不只是重复 E3a 的 section cue。

代价是 E3a 在 `000036` 上的 Hybrid@5 增益没有保留，Semantic@20 gated hit 也从
E3a 的 `0.8571` 回落到 `0.7857`。更重要的是，context-token ratio 虽从 E3a 的
`1.4185` 改善到 `1.3690`，仍明显高于 gate。当前主要阻塞已经从 stable ranking
loss 收敛为 context selection 偏向较长 chunks。

## Decision

E3b `valid: true`、`promoted: false`。按照冻结协议，不运行 50-case direction
confirmation，也不把 E3a 或 E3b 设为默认 representation。

E3 阶段到此结束。下一项实验应把 ranking representation 固定为 E3b，只改变
context selection，例如加入明确 token budget；该实验必须使用新的 variant、manifest
和 artifacts，不能回写本轮结果。
