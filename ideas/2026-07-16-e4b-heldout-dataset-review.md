# E4b Held-out Dataset Review

- 日期：2026-07-16
- 状态：accepted-dataset
- Generation commit：`66dea16a416848e1e0e69d1997ab23a6a9615a2f`
- Raw artifacts：[E4b held-out generation](../benchmarks/results/heldout-datasets/E4b-heldout-generation-66dea16)
- Reviewed dataset：[e4b_heldout_confirmation.jsonl](../examples/eval/e4b_heldout_confirmation.jsonl)
- Reviewed manifest：[e4b_heldout_confirmation.manifest.json](../examples/eval/e4b_heldout_confirmation.manifest.json)

## Generation Result

生成器从 10 个冻结 PDF spans 产生 20 cases。原始报告满足结构 gate：20 个 nonempty
unique queries、0 个 canonical exact duplicates、10 个 span provenance/hash 全部匹配，
`valid: true`。原始 dataset SHA-256 是
`c731717b616af1db3c5b91e39eb52a50584feca342a97fc38336455acabef168`。

## Manual Review

人工逐条检查 query 自然度、reference grounding、明显答案泄漏和同 span 两题重复度。
19 cases 原样通过；`e4b-heldout-000017` 在检索运行前做一次 grounding 修正：

- 原问题要求给出一般 determinant bound；
- 原答案补出了 `c^n n^{n/2}`，但归档 `span.text` 中该公式行被 PDF 提取器漏掉；
- 保留同一个 case ID、evidence span、question type 和 difficulty；
- query/reference 改为 span text 明确包含的 `c = 1, n = 100` 数值比较；
- 原始模型输出仍完整保存在 generation archive，未覆盖或删除。

这个修正只依据 evidence grounding，不观察 Semantic/Hybrid retrieval 结果，也不删除困难
case。20 个 case 的原始 query/answer hashes、19 个 pass 和 1 个 correction 均冻结在
[`e4b_heldout_manual_review.json`](../benchmarks/e4b_heldout_manual_review.json)。

## Frozen Dataset

- Cases：20；
- Unique queries：20；
- Canonical exact duplicates：0；
- Manual passes/corrections：19/1；
- Reviewed dataset SHA-256：
  `b79bbd7d8dfdff5da36673d9df13d388be4774cf8a0f0fca0d8862955624bccb`；
- Finalizer 会验证 generation summary、raw dataset、review manifest、canonical dataset
  和 10 个 span-run artifacts 的 frozen hashes 后再写 reviewed dataset；
- 后续 ranking/packing runner 只能读取 reviewed dataset，不得修改问题、答案或 evidence。

因此该 dataset 可以进入 E4b held-out confirmation runner 实现与正式运行；此结论不包含
任何 ranking、fragment retention 或 answer-quality claim。
