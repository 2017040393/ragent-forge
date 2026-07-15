# E3a Retrieval Screening Conclusion

- 日期：2026-07-15
- Variant：`E3a-pdf-section`
- 状态：rejected
- Protocol：[Retrieval Representation Screening Protocol](2026-07-14-retrieval-screening-protocol.md)
- Specification：[E3 PDF representations](2026-07-15-e3-pdf-embedding-representations.md)
- Artifacts：[E3a screen result](../benchmarks/results/screens/E3a-pdf-section-2930f68)

## 实验边界

E3a 保留 E2 的 `instructed_query_v1`，只改变 PDF document embedding input：
`cleaned_pdf_section_text_v1`。它对 PDF 做确定性 layout cleaning，并从 chapter、
section 和编号 heading 生成可传播的 section cue；Definition/Theorem/Exercise 只作为
当前 chunk 的局部 cue。Markdown chunks 继续使用 E2 的 structured representation。

Chunk content fingerprint、chunk boundaries、完整 corpus、embedding model、retrieval
matrix、16-case slice 和 promotion gates 均未改变。E2 instructed query cache 被复用，
本轮为 64 hits、0 misses。

## 结果

| Configuration | E0 full hit | E3a full hit | E0 gated hit | E3a gated hit |
|---|---:|---:|---:|---:|
| Semantic@5 | 0.4375 | 0.6250 | 0.4286 | 0.6429 |
| Semantic@20 | 0.5625 | 0.8125 | 0.5714 | 0.8571 |
| Hybrid@5 | 0.5625 | 0.6875 | 0.6429 | 0.7143 |
| Hybrid@20 | 0.7500 | 0.8125 | 0.7143 | 0.7857 |

通过的 gates：

- 没有新增 `missed_source`；
- challenge/hard cases 新增 4 个 Semantic@5 hit；
- gated Hybrid@5 增加 1 个 hit；
- evidence mapping coverage 保持 1.0。

失败的 gates：

- stable Semantic@5 丢失 `000016`；
- selected context tokens ratio 为 `1.4185`，超过 `1.10`。

## Interpretation

E3a 是目前最有方向性的 document-side 改动。相对 E2，section cue 让
Semantic@5 从 `0.5000` 提升到 `0.6250`，Semantic@20 从 `0.6250` 提升到 `0.8125`，
并把 Hybrid@5 gated hit 提升到 `0.7143`。这直接支持“长 PDF 内缺少章节语义是
主要 wrong-section 瓶颈”的判断。

但 token ratio 仍然过高，说明更好的 dense ranking 还在把较长 chunks 带入最终
选择；`000016` 仍未恢复，说明 section cue 不是完整解决方案。

## Decision

E3a 不晋级，不运行 50-case direction confirmation。继续 E3b：完整复用 E3a 的
section/cleaning 表示和 E2 query，只增加高置信度 formula evidence，观察公式表示
能否补充稳定命中而不继续扩大 context-selection 偏置。
