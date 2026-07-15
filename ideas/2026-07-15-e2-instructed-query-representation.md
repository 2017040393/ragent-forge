# E2 Instructed Query Embedding Representation

- 日期：2026-07-15
- 状态：implemented
- 实验阶段：pre-v0.3 representation experiments
- Parent：[E1 screening conclusion](2026-07-15-e1-screening-conclusion.md)
- Protocol：[Retrieval representation screening protocol](2026-07-14-retrieval-screening-protocol.md)

## 目的

E2 用一个独立受控变量检验 E1 的下降是否来自 document/query embedding input
不对称。E2 复用 E1 的 `structured_document_text_v1` index，只把 query-side input
从 `raw_query_v1` 改为 `instructed_query_v1`。

```text
E1: structured_document_text_v1 -> raw_query_v1
E2: structured_document_text_v1 -> instructed_query_v1
```

E2 不重新 ingest，也不重建 document index。chunk content、document vectors、
embedding model、retrieval modes、limits、case groups 和 promotion gates 必须与 E1
保持一致。

## Canonical Template

`instructed_query_v1` 的 provider input 固定为：

```text
Instruct: Retrieve the document passage that best answers the query. Distinguish the correct section from other passages in the same source document.
Query: <original query>
```

模板规则：

1. `Instruct:` 和 `Query:` 标签、英文标点、大小写和换行必须保持不变。
2. `<original query>` 使用 evaluator 传入的完整 query，不改写、不翻译、不添加
   metadata，也不做额外 strip。
3. query cache key 使用最终 instructed provider input 的 SHA-256，而不是 raw query
   的 SHA-256。
4. cache 必须记录 `query_representation=instructed_query_v1`，不得读取或混用 E0/E1
   的 `raw_query_v1` cache。
5. 同一 represented query 在四组 screen configurations 中只调用 provider 一次。

## Fixed Boundary

E2 复用以下 E1 workspace state：

| Item | Value |
|---|---|
| Workspace | `.ragent/baselines/E1-structured-document-a43695b` |
| Workspace build commit | `a43695b` |
| Snapshot | `snapshot-20260715T033549Z-a1f9077f` |
| Chunk count | 1744 |
| Chunk content fingerprint | `a99c38a3...3278a` |
| Index input fingerprint | `46d8a7b0...c24bff` |
| Document representation | `structured_document_text_v1` |

预期 query cache 行为是 16 misses、48 hits。E2 不使用
`--query-cache-source`，因为 raw 和 instructed query representations 不兼容。

## Decision Rule

E2 必须相对冻结的 E0 parent 通过全部 promotion gates。仅仅高于 E1 不构成晋级。

- 通过：进入 50-case semantic/hybrid direction confirmation。
- 未通过：停止当前 `structured_document_text_v1` 路线，不进入 50-case 或正式
  36-trial matrix；后续先重新设计 document representation 或 PDF section semantics。

Screen latency 仍是 diagnostic data，不能用来形成 release latency claim。

最终结果见 [E2 screening conclusion](2026-07-15-e2-screening-conclusion.md)。
