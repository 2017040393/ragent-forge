# v0.2 Retrieval Baseline for v0.3

- 日期：2026-07-11
- 最后更新：2026-07-12
- 状态：baseline-frozen
- Pipeline / corpus 版本：`fc81d2a`
- 相关版本：v0.2 baseline，服务于 v0.3 research
- 研究顺序：[v0.3 retrieval 研究顺序](2026-07-11-v0-3-retrieval-research-order.md)
- Versioned cases：
  [`v0_2_retrieval_baseline.generated.jsonl`](../examples/eval/v0_2_retrieval_baseline.generated.jsonl)
- Dataset manifest：
  [`v0_2_retrieval_baseline.generated.manifest.json`](../examples/eval/v0_2_retrieval_baseline.generated.manifest.json)

## 目的与 Snapshot Boundary

在改变 retrieval unit construction、embedding usage、hybrid fusion 或 indexing
之前，记录当前 v0.2 pipeline 在 `examples/knowledge` corpus 上的可复现表现。

本页已经取代早期的 3-document / 796-chunk baseline。新增的线性代数 PDF 已经
完成 ingest、chunk 和 index build；所有主指标也已经改用当前 generation model
生成并扩充到 50 条的 span-grounded cases。旧数字不再代表当前 corpus。

这仍然不是正式 benchmark claim。50 条 AI-generated cases 适合固定当前行为、暴露
失败模式和支持 v0.3 对照实验，但不足以证明通用检索质量。

## 运行配置

| 配置 | 值 |
|---|---|
| Workspace | `.ragent` |
| Corpus | `examples/knowledge` |
| Chunk size | 1000 Python Unicode code points |
| Chunk overlap | 0 |
| Retrieval modes | lexical、BM25、semantic、hybrid |
| Evaluation limit | top-5；同一报告计算 Hit@1/3/5 |
| Semantic similarity | cosine similarity |
| Hybrid fusion | BM25 + semantic weighted RRF |
| Embedding provider | OpenAI-compatible endpoint at `api.siliconflow.cn` |
| Embedding model | `Qwen/Qwen3-VL-Embedding-8B` |
| Embedding dimensions | 4096 |
| Embedding batch size | 8 |
| Generation provider | `openai_responses` at `api.aijws.com` |
| Generation model | `gpt-5.6-luna` |
| Generation reasoning effort | `medium` |
| Generation temperature | 0.2 |

API keys 只保存在 ignored `.ragent/config.toml`，没有写入 versioned dataset、
manifest、baseline 文档或报告摘要。

## Corpus 与 Ingestion

输入文件：

- `examples/knowledge/agentic_rag.md`
- `examples/knowledge/rag_basics.md`
- `examples/knowledge/High-Dimensional Probability_ An Introduction wit.pdf`
- `examples/knowledge/linear_algebra_done_right_4e.pdf`

Ingestion 与 chunk 结果：

| 项目 | 数值 |
|---|---:|
| Documents | 4 |
| Total chunks | 1744 |
| High-dimensional probability PDF chunks | 794 |
| Linear algebra PDF chunks | 948 |
| `rag_basics.md` chunks | 1 |
| `agentic_rag.md` chunks | 1 |
| Minimum chunk code points | 2 |
| Maximum chunk code points | 1000 |
| Average chunk code points | 800.93 |
| Chunks under 200 code points | 146 |
| Chunks over 1000 code points | 0 |

字符统计使用与 Python `len(text)` 一致的 Unicode code point 口径。PowerShell
`.Length` 统计 UTF-16 code units，会把数学 PDF 中的 supplementary-plane symbols
计为两个 unit，因此不能直接用于判断 chunk 是否超过 `chunk_size=1000`。

PDF extraction：

| 项目 | 数值 |
|---|---:|
| PDF files | 2 |
| Pages seen | 709 |
| Pages with text | 703 |
| Empty pages | 5 |
| Structured blocks across corpus | 716 |
| Tables extracted | 7 |
| Possible formula blocks | 682 |
| Possible formula lines | 5526 |
| Reading-order fallback pages | 0 |
| Suspected headers filtered | 140 |
| Suspected footers filtered | 440 |

Ingestion 记录了 21 个 `table_empty` warnings 和 5 个 `empty_page` warnings；它们
没有阻止 ingest。当前统计同时暴露出 146 个不足 200 code points 的短 chunks，
应作为 v0.3 retrieval-unit experiments 的一个观察维度。

## Semantic Index

| 项目 | 数值 |
|---|---:|
| Indexed chunks | 1744 |
| Embedding dimensions | 4096 |
| Index JSONL size | 162,728,324 bytes，约 155.19 MiB |
| Full build elapsed time | 约 514.7 seconds |

Provider compatibility observation：request batch size 为 32 时，provider 返回的
vector indexes 是 `0..7` 重复四次，而不是 OpenAI-compatible 的 `0..31`。项目
客户端会拒绝这种响应；将 ignored local config 的 batch size 固定为 8 后，1744
个 chunk 全量 index build 成功。

155.19 MiB 的 4096-dimensional JSONL index 只覆盖 1744 个 chunks，已经说明
index representation、加载策略和 query-time exact scan 是 v0.3 的实际效率问题。

## AI-Generated Eval Dataset

canonical dataset 保留原始 20 条 cases，并从两个 PDF 的未覆盖位置新增 15 个
evidence spans、每个生成 2 条问题，形成 50 条 cases。新增位置限制在每本 PDF
可用 spans 的 5% 到 95% 范围内，再通过 farthest-point fill 补齐，避免样本集中在
封面、目录、索引或少数相邻章节。

所有 source paths 都使用 repository-relative 格式，dataset 与 manifest 纳入版本
控制，以便换电脑后继续复现实验。

| 项目 | 数值 |
|---|---|
| Dataset | `examples/eval/v0_2_retrieval_baseline.generated.jsonl` |
| Manifest | `examples/eval/v0_2_retrieval_baseline.generated.manifest.json` |
| Cases | 50 |
| Base cases retained | 20 |
| Newly generated cases | 30 |
| Selected evidence spans | 25 |
| Questions per span | 2 |
| Generation errors | 0 |
| Generation method | `llm_synthetic_span_v0.2` |
| Selection strategy | `source_balanced_farthest_fill_v2` |
| Canonical dataset SHA256 | `4ae0d8e3c7be3686811cf481ff98b206179796ed3e38eeada3b5023e17631972` |

Source coverage：

| Source | Selected spans / cases | Selected pages |
|---|---:|---|
| `agentic_rag.md` | 1 / 2 | N/A |
| High-dimensional probability PDF | 11 / 22 | 35、53、72、90、109、131、153、175、198、233、269 |
| Linear algebra PDF | 12 / 24 | 46、70、96、121、147、177、208、238、269、293、318、369 |
| `rag_basics.md` | 1 / 2 | N/A |

Question distribution：

- Type：23 factual、15 reasoning、9 how-to、3 comparison。
- Difficulty：27 easy、21 medium、2 hard。

限制：

- cases 是 AI-generated，新增 30 条已做问题、答案与 evidence 的人工抽检，但仍需
  在后续版本加入 hard negatives 和独立 reviewer 标注。
- 两个 Markdown 各只有 2 条 cases，来源级比例不能视为稳定估计。
- 当前只有 2 条 hard case，也没有系统覆盖 ambiguous、multi-source 和 adversarial
  query。
- 一个 evidence span 可能映射到多个当前 chunks；因此 Recall@5 衡量命中的 mapped
  chunks 比例，而 Hit@5 只要求至少命中一个 mapped chunk。

## 50-Case Retrieval 结果

本次 compare 按 `lexical,bm25,semantic,hybrid` 顺序执行一次，使用 top-5，全程
没有 provider 失败。完整运行耗时 987.3 seconds。检索排序在此前 20-case 三轮中
已经表现为确定性；扩容后先冻结一轮 50-case 质量结果，避免为重复 dense queries
额外发起两轮外部 embedding 调用。

质量指标：

| Mode | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | Failures / 50 |
|---|---:|---:|---:|---:|---:|---:|
| lexical | 0.1400 | 0.1600 | 0.1800 | 0.1000 | 0.1550 | 41 |
| BM25 | 0.4600 | 0.6800 | 0.7200 | 0.4200 | 0.5557 | 14 |
| semantic | 0.1800 | 0.2600 | 0.3000 | 0.1733 | 0.2280 | 35 |
| hybrid | 0.3000 | 0.5400 | 0.6400 | 0.3600 | 0.4263 | 18 |

端到端检索延迟与 context size：

| Mode | Avg latency ms | Avg context chars | Avg estimated tokens |
|---|---:|---:|---:|
| lexical | 258.35 | 4960.34 | 1240.20 |
| BM25 | 338.95 | 4511.24 | 1128.06 |
| semantic | 9722.91 | 1419.34 | 355.26 |
| hybrid | 9330.38 | 3135.52 | 784.18 |

这里的 semantic/hybrid latency 同时包含 external query embedding、JSONL index
loading、exact cosine scan 和 result materialization，不能直接归因到某一个阶段。

## 按来源与失败类型分析

各来源 Hit@5：

| Mode | Agentic RAG | Probability PDF | Linear algebra PDF | RAG basics |
|---|---:|---:|---:|---:|
| lexical | 1/2 | 4/22 | 4/24 | 0/2 |
| BM25 | 2/2 | 19/22 | 13/24 | 2/2 |
| semantic | 2/2 | 3/22 | 8/24 | 2/2 |
| hybrid | 2/2 | 15/22 | 13/24 | 2/2 |

失败类型：

| Mode | `missed_source` | `wrong_section` |
|---|---:|---:|
| lexical | 10 | 31 |
| BM25 | 7 | 7 |
| semantic | 1 | 34 |
| hybrid | 0 | 18 |

扩容回归检查：

| Mode | 原 20 条 Hit@5 | 新增 30 条 Hit@5 | 新增 30 条 Recall@5 | 新增 30 条 MRR |
|---|---:|---:|---:|---:|
| lexical | 0.2000 | 0.1667 | 0.0944 | 0.1500 |
| BM25 | 0.8000 | 0.6667 | 0.3222 | 0.5250 |
| semantic | 0.4000 | 0.2333 | 0.0944 | 0.1833 |
| hybrid | 0.7500 | 0.5667 | 0.2722 | 0.3778 |

原 20 条在四种模式下的 Hit@1/3/5、Recall@5、MRR 和 failure count 与上一版逐项
一致，说明整体分数下降来自新增的 PDF 难例，而不是原有行为回归。

主要观察：

1. BM25 仍是当前质量最强且延迟最低的主检索方式，Hit@5 为 0.72。
2. BM25 在 probability PDF 上达到 19/22，但在线性代数 PDF 上只有 13/24；泛化
   数学术语容易在两本书之间产生 source confusion。
3. 当前 semantic 并未优于 BM25：对两个 Markdown 是 4/4，但对 probability PDF
   只有 3/22，对 linear algebra PDF 是 8/24。
4. Semantic 的 35 个失败中有 34 个是 `wrong_section`，进一步确认 dense path 的
   主要问题是长 PDF 内的 section-level 定位。
5. Hybrid 消除了全部 `missed_source`，但仍有 18 个 `wrong_section`，最终 Hit@5
   为 0.64，略低于纯 BM25。
6. 失败结构支持 v0.3 先研究 retrieval unit / chunk formation：当前主要问题是
   长文档内的 section-level 定位，不只是候选文档召回。
7. Dense path 每 case 约 9 到 10 秒，BM25 约 0.34 秒。后续需要拆分 query embedding、
   index loading、vector scan 和 result materialization 的计时，并考虑 query
   embedding cache，否则 chunking 实验迭代成本过高。

## Artifacts

Versioned：

- `examples/eval/v0_2_retrieval_baseline.generated.jsonl`
- `examples/eval/v0_2_retrieval_baseline.generated.manifest.json`
- `ideas/2026-07-11-v0-2-retrieval-baseline.md`

Ignored local workspace：

- `.ragent/ingest/latest_summary.json`
- `.ragent/index/vector_index.jsonl`
- `.ragent/index/vector_index_manifest.json`
- `.ragent/eval/v0_2_retrieval_baseline_50.generation_errors.json`
- `.ragent/eval/v0_2_baseline_50_compare_gpt-5.6-luna.json`

## 评估复现命令

```powershell
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent index build --workspace .ragent
uv run ragent eval compare --workspace .ragent --cases examples/eval/v0_2_retrieval_baseline.generated.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 5
```

当前 50-case baseline 使用一次完整 compare；后续 retrieval-unit experiments 应
复用同一个 dataset，并至少运行一次完整 compare。固定 baseline 对比时不要重新
生成 cases；重新调用 generation model 会创建一个新的 dataset snapshot，而不是
复现当前哈希。

## v0.3 下一步

1. 为 evaluator 增加 precision@k、nDCG@k、latency p50/p95、mapping coverage、
   evidence density 和 duplicate-context metrics。
2. 保持当前 retrieval 配置不变，先比较 structure-aware、token-aware、section-aware
   等 retrieval-unit strategies。
3. 对 146 个短 chunks、PDF formula/table blocks 和 evidence 跨 chunk 的情况做
   failure analysis。
4. 固定选出的 retrieval-unit strategy 后，再比较 sparse、dense、hybrid、reranking
   和 embedding model。
5. 将 query embedding、index loading、vector scan、fusion 与 result materialization
   分段计时，再决定 ANN index 或 vector database 是否必要。
