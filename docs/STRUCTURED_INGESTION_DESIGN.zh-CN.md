# v0.1-alpha-1 Unified Structured Ingestion Foundation

> 语言: [English](STRUCTURED_INGESTION_DESIGN.md) | 中文

v0.1-alpha-1 structured ingestion milestone 为本地知识文件加入了统一的 structured
ingestion foundation。本文保留 milestone 的设计理由，但不声称存在一个单独发布的 tag。

相关 milestone 文档：

- [RELEASE_NOTES_V0_1_ALPHA_1.zh-CN.md](RELEASE_NOTES_V0_1_ALPHA_1.zh-CN.md)
- [STRUCTURED_INGESTION_DEMO.zh-CN.md](STRUCTURED_INGESTION_DEMO.zh-CN.md)

## Why This Exists

PDF ingestion 引入了 structured path，因为 page text、tables、reading order、
formulas 和 extraction warnings 需要比普通字符窗口更多的上下文。Markdown 和 TXT
此前仍使用 plain text chunking，这让 ingestion architecture 分裂成两套模型。

v0.1-alpha-1 foundation 让 `DocumentBlock` 成为 supported local formats 的共同中间表示：

```text
Markdown -> Document + DocumentBlock[] -> BlockChunker -> DocumentChunk[]
TXT      -> Document + DocumentBlock[] -> BlockChunker -> DocumentChunk[]
PDF      -> Document + DocumentBlock[] -> BlockChunker -> DocumentChunk[]
```

Ingestion 之后 retrieval 仍是 chunk-based。Search、ask、indexing 和 trace flows 继续消费
`DocumentChunk` records，不需要知道由哪个 loader 生成。

## Loader Layer

`core.ingestion.structured_loader` 是 format dispatch layer。它拥有 supported extension
registry，并把每个文件路由到 structured loader，返回 `StructuredLoadResult`：

- `document`：source document payload 和 document-level metadata。
- `blocks`：有序 `DocumentBlock` records。
- `metadata`：loader-level summary metadata。
- `warnings`：可选 loader warnings，目前由 PDF 使用。

这让 `IngestService` 对格式保持无感。新增另一种本地 document type 应该意味着添加
loader 并注册扩展名，而不是给 service 再加一条 chunking branch。

## Markdown Blocks

Markdown 使用 deterministic、line-based parsing，而不是完整 Markdown AST。目标是有用的
retrieval structure，不是 CommonMark compliance。

支持的 lightweight block types：

- `heading`：从 `#` 到 `######` 的 ATX headings。
- `paragraph`：连续非空且不属于其他 supported block 的行。
- `table`：带 separator row 的简单 pipe tables。
- `list`：连续 unordered 或 ordered list lines。
- `code`：使用 backtick 或 tilde fences 的 fenced code blocks。
- `blockquote`：以 `>` 开头的连续 quote lines。

Markdown block metadata 包括：

- `media_type: text/markdown`
- `start_char` 和 `end_char`
- `section_title`
- `heading_path`
- block-specific fields，例如可用时的 `heading_level`、`serialization` 或
  `code_language`

Markdown tables 默认保持 standalone chunks，因此 table text 不会被合并进周围 paragraphs。

## TXT Blocks

TXT ingestion 通过空行分割 paragraph blocks。每个 block 使用：

- `media_type: text/plain`
- `block_type: paragraph`
- `start_char` 和 `end_char`

较长 TXT paragraphs 可以保持为一个 block。`BlockChunker` 会按照配置的 chunk size 和
overlap 处理 oversized non-table blocks。

## PDF Blocks

PDF 保留 PDF ingestion milestones 中加入的 structured behavior。PDF chunks 继续保留：

- `media_type: application/pdf`
- `page_start` 和 `page_end`
- `table_indices`
- reading order metadata
- table captions 和 table context
- possible formula metadata
- header/footer filtering metadata
- extraction warnings

PDF-only fields 不会添加到 Markdown 或 TXT chunks。

## Chunk Metadata

`BlockChunker` 现在从收到的 blocks 构建 common metadata：

- `source_path`
- `media_type`
- `block_types`
- 当 chunk 只有一种 block type 时的 `block_type`

对于 Markdown 和 TXT，character ranges 会跨组合 blocks 聚合。对于被切分的 oversized
text blocks，每个 chunk 会获得其窗口对应的 character range。对于 Markdown，会保留第一个
有意义的 `section_title` 和 `heading_path`。

对于 PDF，现有 page/table/quality metadata 只会在 chunk media type 为 `application/pdf`
时加入。

## Display Behavior

Compact source labels 保持稳定：

- Markdown：`rag.md`
- TXT：`notes.txt`
- PDF：`paper.pdf p.7 table 2`

Inspector views 可以显示 structured source metadata：

- Markdown：type、section、heading path、block type。
- TXT：type 和 block type。
- PDF：现有 page、table、reading order、quality 和 warning details。
