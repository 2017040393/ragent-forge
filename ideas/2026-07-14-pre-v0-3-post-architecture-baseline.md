# Pre-v0.3 Post-Architecture Retrieval Baseline

- 日期：2026-07-14
- 状态：baseline-frozen
- 相关版本：architecture convergence 后、v0.3 retrieval experiments 前
- Benchmark manifest：
  [`retrieval_baseline_manifest.json`](../benchmarks/retrieval_baseline_manifest.json)
- Machine-readable summary：
  [`summary.json`](../benchmarks/results/pre-v0.3-20260714-c410e2e/summary.json)
- 完整 artifacts：
  [`pre-v0.3-20260714-c410e2e`](../benchmarks/results/pre-v0.3-20260714-c410e2e)
- 历史对照：
  [v0.2 retrieval baseline for v0.3](2026-07-11-v0-2-retrieval-baseline.md)
- 研究顺序：
  [v0.3 retrieval 研究顺序](2026-07-11-v0-3-retrieval-research-order.md)

## 冻结边界

这份 baseline 冻结 architecture convergence 后、任何 v0.3 retrieval quality
experiment 前的 document retrieval 行为。它不是通用 benchmark claim，也不评估
生成答案质量；它用于判断后续 retrieval-unit、candidate retrieval、ranking、context
selection 和 indexing 实验是否真正改善了当前系统。

正式矩阵为 4 modes x 3 requested limits x 3 isolated trials，共 36 个 trial：

- modes：`lexical`、`bm25`、`semantic`、`hybrid`；
- requested limits：5、10、20；
- 每个 trial 的第一个 query 是 cold sample，其余 49 个 query 是 warm samples；
- 每个 mode、limit 和 trial 使用独立 runtime，不跨配置共享 prepared state；
- `lexical` 和 `bm25` 要求三轮 ranking fingerprint 完全一致；
- `semantic` 和 `hybrid` 保留三轮全部 fingerprints，并要求每项 quality metric 的
  spread 不超过 manifest 中预先冻结的 `0.05`。

不能从 top-20 结果截取 top-5 或 top-10 代替独立运行。Requested limit 会影响
candidate depth 和 hybrid fusion，因此三个 limits 是三组不同配置。

## Provenance

| 项目 | 冻结值 |
|---|---|
| Benchmark | `pre-v0.3-post-architecture-convergence` |
| Measured at | `2026-07-14T03:59:06.220051+00:00` |
| Architecture convergence commit | `6fa910c` |
| Workspace build commit | `ca029f912190bbf1dd41f0b0905bdfef592fb459` |
| Trial commits | `c410e2e2629f0009edbee811312ac6122bc2c978`、`ca3900619573498b2cc5bff627c36092787f398a` |
| Summary commit | `ca3900619573498b2cc5bff627c36092787f398a` |
| Workspace snapshot | `snapshot-20260714T024249Z-60c124c1` |
| Dataset | 50 cases，`b92b9b0c...`（`text_lf`） |
| Corpus | 4 documents，1744 chunks |
| Index | 1744 vectors x 4096 dimensions |
| Embedding | `Qwen/Qwen3-VL-Embedding-8B`，batch 8，timeout 60 s |
| Runtime | CPython 3.12.11，Windows 11，AMD64 |
| Manifest SHA256 | `2424aceffb22b95a702160de1dc3048200cf7646b1d90b4c826d32c86863f769` |
| Result | `passed: true` |

输出目录名保留第一次正式运行的 `c410e2e` 标识。长矩阵中断后使用 `--resume`
完成，因此 individual trials 来自两个 clean commits；`summary.json` 已逐项记录并
验证这些 commits，没有覆盖已完成 trials。

旧 baseline 文档记录的 dataset SHA256 `4ae0d8e3...` 是 Windows CRLF 原始字节
哈希。新 manifest 为跨平台复现使用 LF 规范化后的 `b92b9b0c...`。当前文件仍由
同样的 50 行、同样的 cases 构成，二者不是不同 dataset。

## Quality Results

以下均为三轮平均值。每一行都由该行的 requested limit 独立执行，不是从更大的
result list 截断得出。

| Mode | k | Hit | Recall | Precision | nDCG | MRR | Avg selected tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| lexical | 5 | 0.1800 | 0.1000 | 0.0440 | 0.1086 | 0.1550 | 1240.20 |
| lexical | 10 | 0.2800 | 0.1533 | 0.0340 | 0.1295 | 0.1673 | 2474.44 |
| lexical | 20 | 0.4000 | 0.2000 | 0.0230 | 0.1444 | 0.1743 | 4941.88 |
| BM25 | 5 | 0.7200 | 0.4200 | 0.1840 | 0.4121 | 0.5557 | 1128.06 |
| BM25 | 10 | 0.7200 | 0.4600 | 0.1020 | 0.4288 | 0.5557 | 2294.10 |
| BM25 | 20 | 0.8000 | 0.5200 | 0.0580 | 0.4484 | 0.5613 | 4567.78 |
| semantic | 5 | 0.2600 | 0.1567 | 0.0520 | 0.1533 | 0.1987 | 317.22 |
| semantic | 10 | 0.3067 | 0.1722 | 0.0307 | 0.1602 | 0.2045 | 597.27 |
| semantic | 20 | 0.3800 | 0.2200 | 0.0210 | 0.1766 | 0.2103 | 1463.95 |
| hybrid | 5 | 0.6400 | 0.3589 | 0.1520 | 0.3292 | 0.4348 | 724.50 |
| hybrid | 10 | 0.7467 | 0.4500 | 0.0987 | 0.3433 | 0.3948 | 1467.43 |
| hybrid | 20 | 0.7867 | 0.5222 | 0.0583 | 0.3650 | 0.3952 | 3076.17 |

当前 BM25 仍是 strongest quality baseline。Hybrid 在 k=20 的 recall 与 BM25
基本相当，但 k=5 precision、nDCG 和 MRR 仍低于 BM25。Semantic 在三个 limits
都明显落后于 BM25，因此 E0/E1 应继续诊断 retrieval representation 和长 PDF
内部的 `wrong_section`，不能把 dense retrieval 的存在本身当作质量提升。

## Cold/Warm Latency

`warm p95` 使用每个配置三轮共 147 个 warm samples 汇总；最后一列同时给出每轮
各自 p95 的最小值和最大值，避免 pooled percentile 隐藏 provider 尾延迟。

| Mode | k | Cold p95 ms | Warm avg ms | Warm p50 ms | Warm p95 ms | Trial p95 range ms |
|---|---:|---:|---:|---:|---:|---:|
| lexical | 5 | 160.86 | 47.10 | 45.13 | 65.09 | 62.65-67.42 |
| lexical | 10 | 157.15 | 47.83 | 46.02 | 65.60 | 64.05-66.76 |
| lexical | 20 | 153.88 | 48.60 | 46.26 | 67.69 | 65.38-69.37 |
| BM25 | 5 | 163.86 | 47.24 | 44.72 | 64.59 | 62.76-72.51 |
| BM25 | 10 | 155.25 | 47.22 | 44.52 | 67.57 | 63.91-70.70 |
| BM25 | 20 | 159.18 | 47.17 | 44.37 | 68.69 | 64.61-70.55 |
| semantic | 5 | 5488.45 | 2007.72 | 1730.27 | 3208.23 | 2761.71-3517.57 |
| semantic | 10 | 4814.54 | 1546.19 | 1505.19 | 1698.50 | 1665.56-1877.37 |
| semantic | 20 | 4636.61 | 1511.35 | 1498.23 | 1660.85 | 1627.40-1680.84 |
| hybrid | 5 | 5070.70 | 1625.91 | 1610.62 | 1827.70 | 1704.01-1890.88 |
| hybrid | 10 | 7466.05 | 1734.13 | 1626.38 | 2712.65 | 1956.23-3080.69 |
| hybrid | 20 | 4982.00 | 1638.79 | 1400.28 | 1978.61 | 1386.72-6390.96 |

Hybrid@20 的 pooled warm p95 是 1978.61 ms，但 trial p95 范围达到
1386.72-6390.96 ms。这说明 external embedding provider 仍有明显网络尾延迟；
后续比较必须继续报告三轮范围，不能只看 pooled p95。

## Dense Variability

Sparse path 是确定性的：每个 lexical/BM25 configuration 的三轮 fingerprint 完全
一致，quality spread 为 0。每个 semantic/hybrid configuration 的三轮 fingerprint
都不同，但全部 quality metric spread 低于预先冻结的 0.05，因此正式结果保留三轮
均值、最小值、最大值和 spread，而不宣称 dense ranking bitwise deterministic。

Release-gate 相关的 dense spread：

| Metric | Average | Min | Max | Spread |
|---|---:|---:|---:|---:|
| Hybrid Recall@20 | 0.5222 | 0.5200 | 0.5267 | 0.0067 |
| Hybrid Precision@5 | 0.1520 | 0.1480 | 0.1560 | 0.0080 |
| Hybrid nDCG@10 | 0.3433 | 0.3409 | 0.3447 | 0.0038 |
| Hybrid MRR@10 | 0.3948 | 0.3903 | 0.4027 | 0.0124 |

一次辅助的 identical-query provider probe 中，两次 embedding 的 cosine similarity
约为 `0.999921`，最大单维绝对差约为 `0.00437`。这与 ranking fingerprints 的
轻微变化一致，但该 probe 不属于 release gate；正式判断以 versioned trial
artifacts 中的 metric spread 为准。

## Historical Comparison

旧报告只执行一次 top-5 compare；新报告执行三轮并区分 cold/warm。下面的 quality
可以直接对照，因为 corpus、chunking 和 50 cases 未变。旧 latency 是单轮 overall
p95，新 latency 是三轮 pooled warm p95，因此只能说明方向，不能解释为严格的
端到端 speedup benchmark。

| Mode@5 | Old Hit | New Hit | Old nDCG | New nDCG | Old MRR | New MRR | Old p95 ms | New warm p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical | 0.1800 | 0.1800 | 0.1086 | 0.1086 | 0.1550 | 0.1550 | 361.30 | 65.09 |
| BM25 | 0.7200 | 0.7200 | 0.4121 | 0.4121 | 0.5557 | 0.5557 | 492.58 | 64.59 |
| semantic | 0.3000 | 0.2600 | 0.1713 | 0.1533 | 0.2280 | 0.1987 | 13392.55 | 3208.23 |
| hybrid | 0.6400 | 0.6400 | 0.3265 | 0.3292 | 0.4263 | 0.4348 | 11774.64 | 1827.70 |

Architecture convergence 后，sparse quality 完全保持，Hybrid@5 quality 也基本保持；
prepared state 复用使 warm latency 显著下降。Semantic quality 比旧单轮结果低，且
三轮结果表明这不是可以忽略的文档转录误差，后续 E0/E1 必须继续调查。

## Frozen Release Gates

Roadmap 规定以 v0.2 hybrid retriever 为 v0.3 初始 baseline。本页把未显式指定的
cutoff 固定为：quality 使用 Hybrid Recall@20、Precision@5、nDCG@10、MRR@10；
latency 和 context cost 使用 Hybrid@5 的 pooled warm p95 与平均 selected tokens。

| Gate input | Frozen baseline |
|---|---:|
| Recall@20 | 0.5222 |
| Precision@5 | 0.1520 |
| nDCG@10 | 0.3433 |
| MRR@10 | 0.3948 |
| Warm p95@5 | 1827.6952 ms |
| Avg selected tokens@5 | 724.50 |

由 roadmap 规则换算后的绝对阈值：

| Metric | Quality-oriented | Efficiency-oriented |
|---|---:|---:|
| Recall@20 | >= 0.5722 | >= 0.5122 |
| Precision@5 | >= 0.2020 | >= 0.1420 |
| nDCG@10 | >= 0.3333 | >= 0.3333 |
| MRR@10 | >= 0.3848 | roadmap 未单列 |
| Warm p95@5 | <= 2741.5428 ms | <= 1462.1562 ms |
| Avg selected tokens@5 | <= 724.50 | <= 615.83 |

这些阈值从现在起用于 v0.3 implementation experiments，不根据后续结果移动。
Declared query-category 或 source-kind slice 仍遵守 roadmap 的 3 个百分点回退上限。

## Artifacts 与复现

Versioned result directory 共 38 个文件、30,471,919 bytes：

- 1 个 resolved `summary.json`；
- 1 个 copied `manifest.json`；
- 36 个完整 trial JSON，位于 `runs/`。

每个 trial 保存 ranking fingerprint、quality metrics、cold/warm latency、stage
timings、cache state 和完整 retrieval evaluation。`summary.json` 保存 runtime、Git、
dataset、corpus、workspace snapshot、index 和三轮聚合结果。Artifacts 不包含 API
key 或 bearer token。

复现命令和 resume 约束见
[`benchmarks/README.md`](../benchmarks/README.md)。新的实验必须使用新的 output
directory，不能覆盖本目录；固定 baseline 对比也不能重新生成 50-case dataset。

## 下一步

Baseline freeze 已完成。按研究顺序进入 E0/E1 representation ablation：保持 corpus、
50-case dataset、retrieval modes 和 release gates 不变，先比较 retrieval unit 的
原始文本与 enriched `embedding_text`，并按 exact-term、paraphrase、formula /
definition、cross-section distractor slices 分析 semantic 的 `wrong_section`。在这一步
得到证据之前，不先调整 RRF 权重、引入 reranker、ANN index 或 vector database。
