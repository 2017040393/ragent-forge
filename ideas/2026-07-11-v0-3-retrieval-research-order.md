# v0.3 Retrieval 研究顺序

- 日期：2026-07-11
- 状态：accepted-direction
- 相关版本：v0.3
- 相关方向：[统一检索入口与 typed sources](2026-07-11-unified-retrieval-typed-sources.md)
- 正式目标：[v0.3 roadmap](../docs/roadmap.zh-CN.md#v03-retrieval-quality-and-efficiency-engineering)

## 触发背景

v0.3 的目标是提升检索效率、召回率和精确度。开始研究具体检索技术之前，需要先
判断项目中的 retrieval unit 如何形成，因为 chunk 的边界、大小、结构和 metadata
会直接影响后续 sparse、dense 与 hybrid retrieval 的上限。

这里把传统的“chunking”扩展为 **retrieval unit construction**：document evidence
需要切分，而 project fact、user note 和 session memory 不一定适合使用固定长度
chunk。

## 当前实现基线

当前 document ingestion 路径是：

```text
Markdown / TXT / PDF
  -> Structured Loader
  -> DocumentBlock[]
  -> BlockChunker
  -> DocumentChunk[]
  -> sparse / dense indexes
```

当前代码具备以下基础：

- `DocumentBlock` 已区分 paragraph、heading、table、formula、list、code、
  blockquote、caption 等 block types。
- `BlockChunker` 默认使用 `chunk_size=1000` 字符和 `chunk_overlap=0`。
- 普通 blocks 会在字符预算内合并；table 单独形成 chunk。
- 超长 block 会按字符窗口切分，overlap 只参与这种超长 block 的滑动切分。
- Chunk metadata 保留 source、字符区间、页码、block type 和表格等结构信息。
- Evidence spans 可以按字符区间或 PDF page overlap 映射到重新生成的 chunks，
  因此同一 eval dataset 可以跨 chunking strategy 复用。
- 当前 semantic retrieval 使用 cosine similarity 做本地精确扫描；hybrid retrieval
  使用 BM25 与 semantic candidates，再通过 weighted RRF 融合。

相关代码：

- [`IngestService`](../src/ragent_forge/app/services/ingest_service.py)
- [`DocumentBlock`](../src/ragent_forge/core/ingestion/document_blocks.py)
- [`BlockChunker`](../src/ragent_forge/core/chunking/block_chunker.py)
- [`GoldChunkMappingService`](../src/ragent_forge/app/services/gold_chunk_mapping_service.py)
- [`SemanticSearchService`](../src/ragent_forge/app/services/semantic_search_service.py)
- [`HybridSearchService`](../src/ragent_forge/app/services/hybrid_search_service.py)

## 核心判断

研究顺序应当是：

```text
冻结 baseline 与补齐指标
  -> retrieval unit construction
  -> candidate retrieval
  -> ranking 与 context selection
  -> indexing 与运行效率
  -> 最终联合验证
```

不能在同一轮实验中同时改变 chunking、embedding model 和 hybrid 参数，否则无法
判断质量变化来自哪一层。

## 0. 冻结 Baseline 与补齐指标

继续复用 v0.2 的 versioned retrieval eval cases，并记录当前 corpus、workspace
配置、chunking 参数、embedding 配置、retrieval limits 与运行环境。

当前 evaluator 已覆盖：

- `hit@k`
- `recall@k`
- `MRR`
- average retrieval latency
- retrieved context characters
- estimated context tokens

v0.3 在比较方案前还需要补充：

- `precision@k`
- `nDCG@k`
- retrieval latency `p50` 与 `p95`
- evidence-span-to-chunk mapping coverage
- retrieved context evidence density
- duplicate 或 overlapping context ratio

这表示 v0.2 eval dataset 可以继续使用，但 evaluator 和报告需要支持 v0.3 的
precision 与 efficiency 目标。

## 1. Retrieval Unit Construction

### Document evidence

需要研究：

- 字符预算与 token 预算的差异。
- heading 是否应与其后 section 内容绑定。
- paragraph、sentence、section 与 page boundaries 如何参与切分。
- table、code、list、formula 是否使用不同的形成策略。
- 超长 block 如何避免从句子、代码或表格中间截断。
- overlap 应固定、按结构产生，还是默认关闭。
- 是否需要 parent-child 或 hierarchical retrieval units。
- chunk IDs 与 chunking strategy version 如何表达。
- 检索后是否合并相邻 chunks，以及合并发生在哪个 pipeline stage。

### Typed memory sources

- `project_fact` 应优先保持为原子事实，而不是按长度切分。
- `user_note` 应保留用户写入时的结构与 provenance。
- `session_memory` 可以按 turn、event、topic 或 summary 形成，具体策略尚未确定。
- 所有 retrieval units 共用一个 retrieval entry point，但保留 source type、
  provenance、authority、freshness 与 lifecycle metadata。

## 2. Candidate Retrieval

固定 retrieval-unit strategy 后，再依次研究：

- lexical token overlap 与中文/英文 tokenization。
- BM25 参数和 sparse candidate depth。
- embedding model、维度、归一化与 cosine similarity。
- BM25 + semantic hybrid retrieval。
- RRF 的 `rrf_k`、sparse/dense weights 与 candidate limits。
- metadata 和 source-type filtering 对召回率的影响。
- exact-term、paraphrase、ambiguous 与 multi-source queries 的表现差异。

本阶段首先关注候选集是否包含正确证据，即 candidate `hit@k` 和 `recall@k`。

## 3. Ranking 与 Context Selection

候选召回之后再研究最终精确度：

- duplicate 与 adjacent chunks 的合并或去重。
- optional reranking。
- source authority、freshness 与 conflict handling。
- evidence coverage 与 source diversity。
- context token budget。
- 多个短证据与单个长 chunk 的取舍。
- context ordering 和 lost-in-the-middle 风险。

本阶段重点观察 `precision@k`、`MRR`、`nDCG@k`、context evidence density 和
selected context tokens。

## 4. Indexing 与运行效率

当前 semantic retrieval 会遍历全部 index records 并逐个计算 cosine similarity。
在测出数据规模与延迟边界后，再研究：

- exact scan 的可接受规模。
- HNSW、IVF 等 approximate nearest-neighbor indexes。
- embedding cache、batching 与并发。
- 增量 index build、更新和删除。
- 是否需要 FAISS、LanceDB、Qdrant 或其他 vector database。

Vector database 选型应由数据规模、filtering、更新、并发和 latency measurements
驱动，而不是作为 v0.3 的默认前置技术。

## 实验纪律

1. 使用当前 chunking 和 retrieval 生成 v0.2 baseline report。
2. 固定 retrieval，分别比较 retrieval-unit strategies。
3. 选出一到两个 unit strategies，避免全参数笛卡尔积。
4. 固定 unit strategy，比较 sparse、dense、hybrid 和后续 ranking 方案。
5. 最后只对少量候选组合做 chunking + retrieval 联合验证。
6. 每次实验保留 manifest、metrics、failure analysis 和可复现配置。

## 当前未决定的技术问题

- token-based、structure-aware 或 hierarchical chunking 的最终实现。
- embedding model 与向量维度。
- query rewriting、query expansion 或 reranking 是否进入 v0.3。
- ANN index 或 vector database 的选择。
- 不同 typed sources 是否共享物理 index。

这些问题应在 baseline 和对应实验结果出现后决定，而不是在研究开始前写死。

## 下一步

当前 v0.2 baseline 已记录在
[v0.2 retrieval baseline for v0.3](2026-07-11-v0-2-retrieval-baseline.md)。下一步
补齐 v0.3 必需的 precision、ranking 与 latency percentile metrics，再开始第一轮
retrieval-unit experiments。
