# E2 Retrieval Screening Conclusion

- 日期：2026-07-15
- Variant：`E2-instructed-query`
- 状态：rejected
- Protocol：[Retrieval Representation Screening Protocol](2026-07-14-retrieval-screening-protocol.md)
- Representation：[E2 instructed query representation](2026-07-15-e2-instructed-query-representation.md)
- Artifacts：[E2 screen result](../benchmarks/results/screens/E2-instructed-query-6c453bb)

## 实验边界

E2 复用 E1 的 `structured_document_text_v1` document index，只把 query embedding
input 从 `raw_query_v1` 改为 `instructed_query_v1`。完整 4-document/1744-chunk
corpus、embedding model、retrieval modes、limits、16-case slice 和 promotion gates
均未改变。

Query cache 使用 represented query 的 SHA-256 key，产生 16 misses、48 hits，没有
读取 E0/E1 的 raw-query cache。Chunk content 和 index input fingerprints 与 E1
完全相同。

## 结果

| Configuration | E0 full hit | E1 full hit | E2 full hit | E0 gated hit | E2 gated hit |
|---|---:|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.3750 | 0.5000 | 0.4286 | 0.5000 |
| Semantic@20 | 0.5625 | 0.5000 | 0.6250 | 0.5714 | 0.5714 |
| Hybrid@5 | 0.5625 | 0.6250 | 0.5625 | 0.6429 | 0.5714 |
| Hybrid@20 | 0.7500 | 0.6875 | 0.6875 | 0.7143 | 0.6429 |

Passed gates：

- 没有新增 `missed_source`；
- challenge cases 新增 3 个 Semantic@5 hit、2 个 Semantic@20 hit；
- evidence mapping coverage 保持 1.0。

Failed gates：

- stable Semantic@5 仍丢失 `000016` 和 `000022`；
- gated Hybrid@5 减少 1 个 hit；
- gated Hybrid@5 selected-token ratio 为 `1.3378`，超过 `1.10`。

## Interpretation

E2 相对 E1 恢复了 `000013`，并把 full Semantic@5 从 `0.3750` 提高到 `0.5000`，
说明 instructed query 能缓解一部分 document/query representation mismatch。它没有
恢复 PDF 内部的两个 stable cases，也没有让 dense signal 在 Hybrid 中产生净收益。
剩余问题仍主要位于 document-side PDF/section representation，而不是 query input
是否带 retrieval instruction。

## Decision

E2 不晋级、不运行 50-case direction confirmation，也不进入正式 36-trial matrix。
后续 E3 应保留 `instructed_query_v1`，只改变 PDF/formula document representation，
继续使用同一组 screening gates。
