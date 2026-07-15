# E3 PDF Embedding Representations

- 日期：2026-07-15
- 状态：accepted-direction
- 实验阶段：pre-v0.3 representation experiments
- Parent：[E2 screening conclusion](2026-07-15-e2-screening-conclusion.md)
- Protocol：[Retrieval representation screening protocol](2026-07-14-retrieval-screening-protocol.md)

## 目的

E3 保留 E2 的 `instructed_query_v1`，只改变 document-side PDF embedding input。
它不修改 `chunk.text`、chunk boundaries、BM25 input、embedding model、hybrid 参数、
完整 corpus、screen cases 或 promotion gates。

为避免把多个 PDF heuristic 混成一个不可解释的结果，E3 分成两个顺序实验：

```text
E3a  cleaned_pdf_section_text_v1 -> instructed_query_v1
E3b  cleaned_pdf_formula_text_v1 -> instructed_query_v1
```

E3b 严格建立在 E3a 上，只新增高置信度 formula representation。

## E3a: PDF Layout And Section Representation

非 PDF chunks 继续使用 `structured_document_text_v1`。PDF chunks 使用与 E1 相同
的字段顺序，但 `Section` 由确定性 section tracker 生成，`Content` 使用仅供 embedding
的清洗文本：

```text
Document title: <title>
Source: <source filename>
Section: <local or propagated section>
Page: <page range>
Block types: <block types>
Signals: <signals>
Content:
<cleaned PDF text>
```

### Section Rules

每个 PDF document 按原 chunk 顺序独立维护 current section：

1. 现有非空 `heading_path` 或 `section_title` 优先。
2. 从原始 chunk lines 识别高置信度 local headings：
   `Chapter/Part/Section`、带层级编号的 prose heading、显式
   `definition/example/theorem/lemma/proposition/corollary/exercise`，以及页首的
   chapter number + title-case heading。
3. 一个 chunk 最多保留 3 个去重 local headings，按出现顺序用 ` | ` 连接。
4. 有 local headings 时，最后一个 heading 成为该 document 后续 chunks 的 current
   section；没有 local heading 时复用 current section。
5. 仍然无法确定时使用 `unknown`。不同 document 之间禁止传播 section state。

Heading detector 必须保守：纯页码、公式编号、完整 prose sentence、页眉和页脚不能
成为 section。

### PDF Cleaning Rules

清洗只作用于 embedding input，顺序固定为：

1. 标准化 CRLF/CR 为 LF，并移除 soft hyphen。
2. 删除与 metadata `header_footer_candidates` 完全匹配的 lines。
3. 删除孤立的 1-4 位页码 lines。
4. 删除无法恢复字符含义的 `(cid:<number>)` extraction markers，不猜测原字符。
5. 只在 ASCII 字母断词场景中连接 `word-\ncontinuation`。
6. 将剩余 layout whitespace 折叠为单个空格。

原始数学符号和文字顺序保留；E3a 不增加 formula evidence lines。

## E3b: High-confidence Formula Representation

E3b 完整复用 E3a 的 section 和 cleaned content，只在 `Signals` 与 `Content` 之间增加：

```text
Formula evidence: <normalized formula 1> | <normalized formula 2> | <normalized formula 3>
```

无高置信度公式时使用 `Formula evidence: none`。

Formula candidates 只来自当前 chunk metadata 中的 `possible_formula_lines`，并满足：

1. 清洗后长度为 4-240 characters。
2. 包含明确 equation、inequality、set、integral、sum、norm 或 metric signal。
3. 清洗后同时包含数字或数学常量，以及字母或数学 identifier。
4. 纯 prose、只含 CID markers、只含页码或单个变量的 lines 被拒绝。
5. 使用 Unicode NFKC 做 mathematical alphanumeric normalization，并把常见
   `pi/lambda`、minus、comparison 和 membership symbols 映射成稳定 ASCII words。
6. 按原出现顺序去重，最多保留 3 条。

Formula normalization 只转换已有字符，不生成解释、不补全缺失公式，也不调用 LLM。

## Invariants And Provenance

每个 E3 variant 必须满足：

1. 4 documents、1744 chunks、chunk IDs 和 chunk-content fingerprint 与 E0-E2
   完全一致。
2. 每个 variant 使用独立 generation snapshot 和完整重建的 4096-dimensional index。
3. vector record `text_hash` 对应实际 E3 embedding input。
4. index manifest 记录精确 representation ID 和 index-input fingerprint。
5. E3a/E3b 都复用 E2 versioned instructed-query cache，不重新生成 query vectors。
6. Markdown chunks 在 E3a 和 E3b 中与 E2 的 document representation 完全相同。
7. 任意 representation error 让 index build 失败，不静默回退到 raw text。

## Decision Rule

E3a 和 E3b 分别运行固定 16-case screen，均相对冻结 E0 parent 判断。E3b 不因高于
E3a 就自动晋级。只有 `valid: true` 且通过全部 promotion gates 的 candidate 才进入
50-case direction confirmation；screen latency 不能用于 release claim。
