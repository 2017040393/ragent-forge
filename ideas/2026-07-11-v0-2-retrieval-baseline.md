# v0.2 Retrieval Baseline for v0.3

- 日期：2026-07-11
- 状态：exploring
- 代码版本：`d35eb44`
- 相关版本：v0.2 baseline，服务于 v0.3 research
- 研究顺序：[v0.3 retrieval 研究顺序](2026-07-11-v0-3-retrieval-research-order.md)

## 目的

在改变 retrieval unit construction、embedding model usage、hybrid fusion 或
indexing 之前，记录当前 v0.2 pipeline 在 `examples/knowledge` corpus 上的可复现
初始表现。

这不是正式 benchmark claim。Case 数量很少，主要用途是固定当前行为、暴露评测
缺口，并为第一轮 chunking experiments 提供对照。

## 运行环境与配置

- Workspace：`.ragent`
- Corpus：`examples/knowledge`
- Cases：`examples/eval/retrieval_cases.jsonl`
- 补充 cases：`examples/eval/synthetic_span_cases.example.jsonl`
- Chunk size：1000 characters
- Chunk overlap：0
- Retrieval modes：lexical、BM25、semantic、hybrid
- Limits：1、3、5
- Semantic similarity：cosine similarity
- Hybrid fusion：BM25 + semantic weighted RRF
- Embedding provider：OpenAI-compatible endpoint at `api.siliconflow.cn`
- Embedding model：`Qwen/Qwen3-VL-Embedding-8B`
- Embedding dimensions：4096
- Embedding batch size：8

API key 保存在 ignored `.ragent/config.toml` 中，没有写入本笔记或 eval artifacts。

## Corpus 与 Ingestion

输入文件：

- `examples/knowledge/agentic_rag.md`
- `examples/knowledge/rag_basics.md`
- `examples/knowledge/High-Dimensional Probability_ An Introduction wit.pdf`

Ingestion 结果：

| 项目 | 数值 |
|---|---:|
| Documents | 3 |
| Total chunks | 796 |
| PDF chunks | 794 |
| `rag_basics.md` chunks | 1 |
| `agentic_rag.md` chunks | 1 |
| Minimum chunk characters | 9 |
| Maximum chunk characters | 1000 |
| Average chunk characters | 804.70 |
| Chunks under 200 characters | 72 |
| Chunks over 1000 characters | 0 |

PDF extraction：

| 项目 | 数值 |
|---|---:|
| Pages seen | 299 |
| Pages with text | 297 |
| Empty pages | 2 |
| Structured blocks across corpus | 306 |
| Tables extracted | 3 |
| Possible formula blocks | 290 |
| Possible formula lines | 2963 |
| Reading-order fallback pages | 0 |
| Suspected headers filtered | 134 |
| Suspected footers filtered | 212 |

已知 PDF warnings：11 个 `table_empty` warnings 和 2 个 `empty_page` warnings。
这些 warnings 没有阻止 ingest。

## Semantic Index

索引构建结果：

| 项目 | 数值 |
|---|---:|
| Indexed chunks | 796 |
| Embedding dimensions | 4096 |
| Index JSONL size | 74,332,166 bytes，约 70.89 MiB |
| Full build elapsed time | 约 228.2 seconds |

Provider compatibility observation：当 request batch size 为 32 时，provider 返回
32 个 vectors，但 indexes 是 `0..7` 重复四次，而不是 OpenAI-compatible 的
`0..31`。项目客户端会正确拒绝这种响应。将 ignored local config 的 batch size
调整为 8 后，796-chunk index 构建成功。

## Main Source-Path Suite

主 suite 使用 3 条手工 cases，并完整运行三次。它们只声明
`expected_source_paths`，没有 `expected_chunk_ids` 或 evidence spans。因此：

- `hit@k`、MRR 和 failures 有效。
- 当前 evaluator 的 `recall@k` 固定为 `0.0`，不能用于比较质量。
- 这 3 条 cases 都针对两个 Markdown 文件，没有 PDF-positive case。

三次运行的质量指标保持一致：

| Mode | Hit@k | MRR | Failures per run |
|---|---:|---:|---:|
| lexical | 0.6667 | 0.6667 | 1 |
| BM25 | 1.0000 | 1.0000 | 0 |
| semantic | 1.0000 | 1.0000 | 0 |
| hybrid | 1.0000 | 1.0000 | 0 on successful runs |

三次运行的平均检索延迟聚合如下：

| Mode | k | Successful runs | Failed runs | Mean ms | Min ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|
| lexical | 1 | 3 | 0 | 104.05 | 89.42 | 121.58 |
| lexical | 3 | 3 | 0 | 100.68 | 95.42 | 110.74 |
| lexical | 5 | 3 | 0 | 103.81 | 89.26 | 120.47 |
| BM25 | 1 | 3 | 0 | 165.33 | 143.76 | 183.28 |
| BM25 | 3 | 3 | 0 | 207.81 | 184.42 | 244.17 |
| BM25 | 5 | 3 | 0 | 168.48 | 163.16 | 176.26 |
| semantic | 1 | 3 | 0 | 5327.61 | 4756.41 | 5757.39 |
| semantic | 3 | 3 | 0 | 5608.33 | 5132.31 | 6032.22 |
| semantic | 5 | 3 | 0 | 5819.11 | 5283.48 | 6402.23 |
| hybrid | 1 | 3 | 0 | 6471.43 | 6060.32 | 6853.84 |
| hybrid | 3 | 3 | 0 | 5326.07 | 5096.43 | 5541.38 |
| hybrid | 5 | 2 | 1 | 5679.47 | 5523.35 | 5835.58 |

第三次 `hybrid@5` 在 query embedding 时出现 provider read timeout，因此只统计
两次成功运行。失败保留在 baseline 中，作为 external embedding reliability 的
观测，不通过立即重跑将其隐藏。

Lexical 的唯一失败是 `case-001`：查询 “What is retrieval augmented generation?”
时 top result 来自概率论 PDF 的 `chunk-0230`，未命中预期的 `rag_basics.md`。

## Span-Grounded Supplemental Suite

为了验证有效的 `recall@k` 路径，额外运行仓库中的 2 条 span-grounded example
cases。该 suite 只运行一次，仅作为 evaluator smoke check。

| Mode | Hit@k | Recall@k | MRR | Failures |
|---|---:|---:|---:|---:|
| lexical | 0.0000 | 0.0000 | 0.0000 | 2 |
| BM25 | 1.0000 | 1.0000 | 1.0000 | 0 |
| semantic | 1.0000 | 1.0000 | 1.0000 | 0 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 0 |

这些 example spans 只覆盖两个很短的 Markdown 文档，不能替代更大、更平衡的
span-grounded eval set。

## 初步判断

1. 在当前 tiny suite 上，BM25、semantic 和 hybrid 都达到质量上限，无法据此
   判断三者的真实质量差异。
2. BM25 明显优于 simple lexical token overlap；lexical 容易被大型无关 PDF 中的
   词项重合干扰。
3. Semantic 与 hybrid 没有在当前 cases 上带来额外质量收益，但平均 latency 比
   BM25 高一个数量级以上。
4. 当前 latency 把 external query embedding、JSONL index loading 和 exact cosine
   scan 混在一起，不能定位具体瓶颈。
5. 4096-dimensional JSONL index 对 796 chunks 已达到约 70.89 MiB，说明 index
   representation 与 loading strategy 是 v0.3 efficiency research 的实际问题。
6. 现有 3 条主 cases 不足以衡量 recall、precision、nDCG、PDF evidence 或不同
   query categories。

## 原始本地 Artifacts

以下文件位于 ignored `.ragent` workspace，不提交到 Git：

- `.ragent/eval/retrieval_compare_20260711T125038Z.json`
- `.ragent/eval/retrieval_compare_20260711T125442Z.json`
- `.ragent/eval/retrieval_compare_20260711T125746Z.json`
- `.ragent/eval/retrieval_compare_20260711T125946Z.json`
- `.ragent/index/vector_index.jsonl`
- `.ragent/index/vector_index_manifest.json`
- `.ragent/ingest/latest_summary.json`

## 复现命令

```powershell
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent index build --workspace .ragent
uv run ragent eval compare --workspace .ragent --cases examples/eval/retrieval_cases.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5
uv run ragent eval compare --workspace .ragent --cases examples/eval/synthetic_span_cases.example.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5
```

主 suite 的 compare 命令需要顺序运行三次。复现前还需要确认 embedding provider
对 batch indexes 的兼容行为；本次运行使用 `batch_size=8`。

## 下一步

1. 扩充 versioned span-grounded cases，加入 PDF-positive、exact-term、paraphrase
   和 harder distractor queries。
2. 在 evaluator 中增加 `precision@k`、`nDCG@k`、latency percentiles、mapping
   coverage 和 duplicate-context metrics。
3. 将 query embedding、index loading、vector scan 与 result materialization 分段
   计时。
4. 固定 retrieval 配置后，开始第一轮 retrieval-unit construction experiments。

