# E1 Structured Document Embedding Representation

- 日期：2026-07-15
- 状态：implemented
- 实验阶段：pre-v0.3 representation experiments
- 研究顺序：[v0.3 retrieval research order](2026-07-11-v0-3-retrieval-research-order.md)
- 筛选协议：[retrieval representation screening protocol](2026-07-14-retrieval-screening-protocol.md)

## 目的

E1 只改变 document-side 的 embedding input，验证显式文档结构是否能帮助
embedding model 在长文档中区分正确 section。embedding model、raw query、chunk
边界、完整 corpus、retrieval modes、requested limits 和 evaluation cases 均保持
与 E0 相同。

E1 的数据流是：

```text
chunk.text + typed metadata
  -> structured_document_text_v1
  -> document embedding

raw query
  -> query embedding
```

`chunk.text` 仍然是最终提供给 LLM 的 evidence context。structured representation
只用于生成和校验 vector index，不修改 chunks 文件中的正文，也不改变最终答案的
上下文内容。

## Canonical Template

每个 chunk 的 embedding input 按以下固定顺序生成。每个字段占一行，字段名和冒号
必须保留；值为空时使用规定的 fallback。

```text
Document title: <title>
Source: <source>
Section: <section>
Page: <page>
Block types: <block_types>
Signals: <signals>
Content:
<original chunk.text>
```

字段规则：

| 字段 | 取值顺序 | 空值 fallback |
| --- | --- | --- |
| `Document title` | metadata `document_title`、`title`；首个有效 heading；source filename stem | source filename stem；再无则 `unknown` |
| `Source` | workspace 中记录的 source path | source filename；再无则 `unknown` |
| `Section` | `heading_path` 的非空部分按 ` > ` 连接；`section_title` | `unknown` |
| `Page` | `page_start`；若有不同的 `page_end` 则 `start-end` | `unknown` |
| `Block types` | `block_types` 或 `block_type` 的去重值，按字典序连接 | `unknown` |
| `Signals` | formula、table 等结构标记，按规定顺序连接 | `none` |

`Source` 必须是 workspace 相对路径或稳定的 source filename，禁止写入机器绝对路径。
`Block types` 和 `Signals` 的排序、去重使用 ASCII 字典序；列表中的空字符串被忽略。
`Page` 只接受整数页码。无法解析的 metadata 不得进入表示文本。

`Signals` 当前只表达结构存在性，不复制 formula 内容：

- 包含 `formula` 或 `possible_formula=true` 时加入 `formula`；
- 包含 `table` 的 block type 时加入 `table`；
- 两者都不存在时使用 `none`。

## Invariants

每个 E1 index build 必须满足：

1. `chunk.text`、chunk count、chunk IDs、chunk-content fingerprint 与 E0 完全一致。
2. E1 query representation 为 `raw_query_v1`，query embedding cache 可以复用 E0。
3. index manifest 记录 `embedding_representation=structured_document_text_v1`。
4. vector record 的 `text_hash` 是实际送入 embedding provider 的 structured text
   hash，而不是 raw `chunk.text` hash。
5. manifest 同时记录 representation version 和 index-input fingerprint，防止
   使用错误表示的旧 index。
6. structured text 由纯函数从 chunk 和 metadata 生成；同一输入在不同机器、不同
   path separator 和不同运行时间下必须产生相同输出。
7. 任意表示生成失败都让 index build 失败，不得静默回退到 raw text。

## Comparison Boundary

E1 screening 复用 E0 的 16-case diagnostic slice、四组 retrieval configuration、
promotion gates、完整 1744-chunk corpus 和 query cache。候选 workspace 必须重新
构建 index，但不得重新 ingest 出不同的 chunk content。E1 通过 screening 后才允许
进入 50-case direction confirmation；screening 本身不产生 release quality 或
latency 结论。

## Non-goals

E1 不做以下改变：

- 不修改 chunking、PDF 清洗、formula normalization 或 source extraction；
- 不改变 raw query，不增加 query instruction；
- 不改变 BM25 文本或最终 hybrid 权重；
- 不把 `embedding_text` 持久化为新的 answer context 字段；
- 不在本实验中引入 reranker、ANN 或 vector database。

