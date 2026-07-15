# E1 Retrieval Screening Conclusion

- 日期：2026-07-15
- Variant：`E1-structured-document`
- 状态：rejected
- Protocol：[Retrieval Representation Screening Protocol](2026-07-14-retrieval-screening-protocol.md)
- Artifacts：[E1 screen result](../benchmarks/results/screens/E1-structured-document-c5e71b2)

## 实验边界

E1 只把 document embedding input 从 `chunk.text` 改为
`structured_document_text_v1`；raw query、embedding model、chunk boundaries、
完整 4-document/1744-chunk corpus、retrieval modes、limits 和 16-case slice 均未改变。
E0 的 16 条 raw query vectors 被复用，query cache 为 64 hits、0 misses。

Chunk content fingerprint 保持 `a99c38a3...3278a`。E1 workspace 使用独立 snapshot
`snapshot-20260715T033549Z-a1f9077f`，所有 index records 都记录
`structured_document_text_v1`，没有混入 raw-text records。

## 结果

| Configuration | E0 full hit | E1 full hit | E0 gated hit | E1 gated hit |
|---|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.3750 | 0.4286 | 0.3571 |
| Semantic@20 | 0.5625 | 0.5000 | 0.5714 | 0.5000 |
| Hybrid@5 | 0.5625 | 0.6250 | 0.6429 | 0.5714 |
| Hybrid@20 | 0.7500 | 0.6875 | 0.7143 | 0.6429 |

Passed gates：

- 没有新增 `missed_source`；
- challenge cases 新增 2 个 Semantic@5 hit；
- evidence mapping coverage 保持 1.0。

Failed gates：

- stable Semantic@5 回退 3 个 case；
- gated Hybrid@5 减少 1 个 hit；
- gated Hybrid@5 selected-token ratio 为 `1.3707`，超过 `1.10`。

## Decision

E1 说明结构字段可以改善部分 wrong-section challenge，但收益不稳定，而且引入了
明显的上下文选择膨胀和 hybrid 回退。因此 E1 不晋级、不运行 50-case direction
confirmation，也不进入正式 36-trial matrix。E1 的代码仍保留为显式可选表示，后续
E2 若继续，必须把 query instruction 作为唯一新增变量并重新通过同一筛选 gates。
