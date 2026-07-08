# PDF Structured Ingestion Design

> 语言: [English](PDF_INGESTION_DESIGN.md) | 中文

## Goal

定义 RAGentForge 计划中的 v0.1.1 PDF Structured Ingestion Foundation。这是一份面向后续
实现工作的设计文档。它不表示 RAGentForge v0.1.0 已经支持 PDF ingestion。

目标是支持 text-based PDFs，并保持 local-first、inspectable、source-grounded、
traceable、eval-friendly，同时兼容后续 retrieval quality 工作。

## Why PDF Support Matters

PDF support 是核心产品方向，不是一个小的可选 importer。许多高价值知识来源都以 PDF
形式分发：

- papers
- technical reports
- textbooks
- manuals
- benchmark reports
- engineering documents

这些来源经常包含 tables、page-level references、formula-like text、captions，以及密集的
multi-page arguments。如果把 PDF ingestion 当作朴素 plain text extraction，search results
会更难检查，retrieval failures 也会被隐藏。v0.1.1 设计应该保留足够结构，让 sources、
traces、TUI Inspector 和未来 retrieval evaluation 可以解释结果来自哪里。

## Version Scope

v0.1.1 应设计支持：

- text-based PDF ingestion
- page-aware metadata
- table extraction
- Markdown table serialization，TSV-like text 作为 fallback
- page/table-aware source display
- PDF block metadata
- PDF extraction warnings
- traceable PDF ingest summaries
- TUI Inspector 中显示 PDF page/table metadata
- 未来 retrieval eval cases 可以引用 PDF source paths，并可选引用 pages

本文档本身不实现 runtime PDF ingestion。未来 v0.1.1 implementation 应在设计接受后，
以小步方式添加代码、依赖和测试。

当前实现说明：v0.1.1 PDF Page + Table Ingestion MVP 是在这份设计之后实现的。较旧的
v0.1.0 文档可能仍把 PDF support 描述为 future work，但当前实现已经包含 text-based PDF
page/table ingestion。

## v0.1.2 PDF Extraction Quality Polish

当前实现把 v0.1.1 作为 page/table ingestion MVP，并通过 v0.1.2 quality-polish milestone
改进 text-based PDFs。v0.1.1 回答第一个 runtime 问题：RAGentForge 能否发现 PDFs、抽取
page text 和 page-local tables、生成 page/table-aware chunks，并显示
`paper.pdf p.7 table 2` 这样的 PDF source labels？v0.1.2 保持该行为，并在 retrieval
看到内容前提升抽取质量。

v0.1.2 不添加 OCR、scanned-PDF support、image text recognition、PDF viewing、PDF
opening、PDF editing 或完整 layout reconstruction。它仍然 local-only 且 deterministic，
使用 `pdfplumber` 暴露的 text layer。如果某页没有可抽取文本，extractor 应继续把它报告为
warning，而不是假装 scanned content 已被处理。

v0.1.2 scope：

- Reading order baseline：当 `pdfplumber` 暴露 words 和 coordinates 时，优先使用
  coordinate/word-aware ordering；保留 page boundaries；避免明显的 two-column
  left/right interleaving；记录 `reading_order_strategy` 和 fallback diagnostics。
- Table caption/context enhancement：检测同页附近的 captions，例如 `Table 2: ...`、
  `TABLE II ...` 或 `表 1 ...`；把安全 captions 前置到 table chunk text；存储
  caption/context metadata，但不合并独立 tables。
- Conservative table/text de-duplication：只移除高置信度、在 paragraph text 中重复出现的
  table row text；永远不删除 table blocks；记录受影响的 lines/pages 数量。
- Formula-like text preservation：保留 PDF text layer 中的 formula-like text，并用
  `possible_formula` 和有界的 `possible_formula_lines` metadata 标记相关 blocks/chunks。
- Header/footer filtering and diagnostics：保守过滤跨页重复的短 header/footer lines，
  保留正文和 page metadata，并记录 suspected headers/footers 的过滤计数。
- Diagnostics propagation：把新的 quality metadata 传递到 ingest summary、ingest trace
  metadata、chunk metadata、search results，以及简洁的 TUI Inspector/source displays。

这些改进有意保持保守。置信度低时，extractor 应优先保留文本并暴露 diagnostics，而不是删除
潜在有用内容。

## Non-Goals

v0.1.1 PDF structured ingestion 应明确排除：

- scanned PDF support
- OCR
- image text recognition
- image formula recognition
- PDF editing
- writing back to PDFs
- PDF viewer
- opening local PDF files
- TUI PDF rendering
- complex cross-page table reconstruction
- chart/image data extraction
- full layout reconstruction
- perfect LaTeX formula recovery
- answer evaluation
- LLM-as-judge
- web UI

Existing Markdown/TXT ingestion、search、Ask、trace、eval 和 TUI behavior 必须继续工作。
PDF metadata 应该是 additive 且 backward-compatible。

## Supported PDF Type

v0.1.1 只支持 text-based PDFs。不支持 scanned PDFs 和 OCR。

Text-based PDFs 是指文本可以被选择、复制，并能从 text layer 中抽取的 PDFs。某个 PDF 仍然
可能有个别页面没有可抽取文本。这些页面应该产生 warnings，而不是静默失败。

Unsupported 或 scanned-like PDFs 不应该被当作成功但为空的 documents。Ingest summary 和 trace
应该说明没有找到 extractable text，未来 CLI output 应清晰暴露该 warning。

## Extraction Requirements

### Page-Aware Text Extraction

PDF ingestion 应逐页处理每个文件。每个 extracted text block 必须保留 source path 和 page
number。展示给用户的 page numbers 应使用 one-based，因为这符合读者引用 PDF 页码的方式。

未来 extractor 至少应记录：

- `source_path`
- `media_type: "application/pdf"`
- one-based `page_number`
- page-local block order
- extracted text
- extraction method
- empty 或 suspicious pages 的 warnings

即使相邻 blocks 后续被合并成更大的 chunks，page breaks 也应该在 metadata 中保留。

### Table Extraction

Table extraction 是 v0.1.1 的核心需求。Tables 不能被静默压平成不可读的 paragraph text。

未来 extractor 应使用所选 PDF library 的 table extraction API 检测 tables、normalize rows
和 cells，并尽可能把 tables 序列化为 Markdown：

```markdown
| Method | Hit@1 | MRR |
|---|---:|---:|
| lexical | 0.67 | 0.78 |
| hybrid | 1.00 | 1.00 |
```

如果某张 table 的 Markdown table serialization 不可靠，fallback 应为 TSV-like text。Fallback
仍应保留 table metadata，这样 source displays 和 evaluation 可以识别结果是 table。

每个 extracted table 应保留：

- one-based page number
- page-local table index
- block type `table`
- row count
- column count
- serialization format，例如 `markdown_table` 或 `tsv`
- extraction warnings（适用时）

当 detected table 为空、没有 usable cells、normalize 后 row widths 不一致，或 serialized form
太短而不实用时，table extraction 应产生 warning。

### Reading Order

v0.1.1 应改进明显的 reading-order problems，但不承诺完美 layout reconstruction。

未来 extractor 应：

- 先在 page level 抽取
- 当 library 提供可靠 layout data 时保留 block-level order
- 当 block coordinates 可用时使用 coordinate-aware sorting
- 尽可能避免明显 two-column interleaving
- 在 metadata 中保留 page breaks
- 把 header/footer detection 留作未来改进

初始实现可以优先选择简单、可审计的 ordering rules，而不是 opaque layout magic。如果 ordering
不确定，extractor 应记录 warning，而不是假装页面被完美重建。

### Formula Text Preservation

Formula handling 很重要，但次于 text 和 table extraction。

当 PDF text layer 中存在 formula-like text 时，未来 extractor 应保留它。它不应有意删除 text
layer 暴露的 mathematical symbols、Greek letters、operators、subscripts、superscripts 或
equation labels。

v0.1.1 不应：

- 对嵌入为图片的 formulas 执行 OCR
- 执行 image formula recognition
- 尝试完整 LaTeX reconstruction
- 保证数学上完美的 formula layout

未来的 block metadata flag，例如 `possible_formula: true`，可以帮助 search、sources 和
Inspector 标识需要人工检查的 blocks。

### Metadata

PDF extraction 应在 chunking 前尽早产生 structured metadata。这让 PDF-specific behavior
集中在 ingestion，而不是散落到 retrieval、Ask、eval 和 TUI code 中。

预期 metadata 包括：

- media type
- source path
- page number 或 page range
- block index
- block type
- table indices
- section title（可恢复时）
- extraction method
- extraction warnings
- optional coordinate/layout metadata

### Warnings and Failure Modes

PDF extraction problems 应可检查，而不是静默。

Warnings 应有足够结构，可以出现在 ingest summaries、traces 和未来 CLI/TUI inspection surfaces。
Warning kinds 可以包括：

- `empty_page`
- `no_extractable_text`
- `table_empty`
- `table_malformed`
- `reading_order_uncertain`
- `unsupported_scanned_pdf`
- `formula_text_suspicious`

一个 warning 至少应包含 source path、可用时的 page number、kind，以及 human-readable message。

## Proposed PDF Block Model

RAGentForge 应在 chunks 之前引入 middle-layer document block model：

```python
@dataclass(frozen=True)
class DocumentBlock:
    source_path: str
    media_type: str
    page_number: int | None
    block_index: int
    block_type: str  # paragraph | heading | table | formula | list | caption | unknown
    text: str
    metadata: dict[str, Any]
```

这个 class 在设计中是概念性的。直到 runtime PDF ingestion 工作开始前，不应实现它。

目标 pipeline 是：

```text
PDF / Markdown / TXT
-> DocumentBlock[]
-> Chunk[]
-> SearchResult
-> Ask sources
-> TUI Inspector
```

Block layer 给每种 document format 一个表达结构的位置。PDF 可以 emit page-aware paragraph
和 table blocks。Markdown 之后可以 emit heading 和 list blocks。TXT 可以 emit paragraph 或
unknown blocks。Chunking 和 source display 随后可以消费共享 shape，而不是在多个 services 中
分散 PDF details 分支。

Example PDF table block metadata：

```json
{
  "table_index": 2,
  "row_count": 3,
  "column_count": 3,
  "serialization": "markdown_table",
  "extraction_method": "pdf_structured",
  "warnings": []
}
```

## Chunking Strategy

PDF chunks 应从有序 `DocumentBlock` records 创建。Chunker 应保持 deterministic，并保留所有
贡献到 chunk 的 blocks 的 source metadata。

Expected PDF chunk metadata：

```json
{
  "source_path": "examples/knowledge/paper.pdf",
  "media_type": "application/pdf",
  "page_start": 3,
  "page_end": 4,
  "block_types": ["paragraph", "table"],
  "section_title": "Retrieval Evaluation",
  "table_indices": [2],
  "extraction_method": "pdf_structured",
  "warnings": []
}
```

Chunking rules：

- Paragraph blocks 可以组合到现有 chunk size target。
- Table blocks 通常应成为 standalone chunks，或锚定到紧邻描述它们的 chunk。
- Table chunk 必须保留 `block_type: table`，或在 `block_types` 中包含 `table`。
- 跨页 chunks 必须设置 `page_start` 和 `page_end`。
- 带 extraction warnings 的 chunks 应继续携带 warnings，让 retrieval results 和 traces 可以解释
  suspicious content。
- Existing Markdown/TXT chunk metadata 必须保持 valid。

第一个实现应避免复杂 cross-page table reconstruction。如果 table 看起来跨页延续，每页本地 table
可以分别抽取，并带上例如 `cross_page_table_possible` 的 warning。

## Source Display Strategy

Search、Ask 和 TUI source displays 应使用 additive PDF labels，同时保持 Markdown/TXT source
labels 不变。

推荐 labels：

```text
paper.pdf p.7
paper.pdf pp.7-8
paper.pdf p.7 table 2
```

Source label rules：

- Compact display 使用 basename，并在 metadata 中保留完整 `source_path`。
- 单页使用 `p.N`。
- Page range 使用 `pp.N-M`。
- 当 primary chunk content 是 table `K`，或 chunk 只有一个 table index 时，追加 `table K`。
- 不要求 local file opening 或 PDF viewer integration。
- 当缺少 page metadata 时，保持 Markdown/TXT behavior 不变。

Ask sources 应包含和 search 相同的 PDF page/table labels，让用户可以把 answer 追溯到 page-level
source。

## TUI Inspector Strategy

TUI Inspector 应以 structured text 显示 PDF source metadata。它不应变成 PDF viewer，也不应打开
local files。

Example Inspector display：

```text
Selected source

rank: 1
source: paper.pdf
type: pdf
page range: 7
block type: table
table: 2
chunk: chunk-0031
score: 0.8321

preview:
| Method | Hit@1 | MRR |
|---|---:|---:|
...
```

Inspector behavior：

- 当 `media_type` 为 `application/pdf` 时显示 `type: pdf`。
- 当 `page_start` 和 `page_end` 存在时显示 page range。
- 当 table metadata 存在时显示 table index。
- 可用时显示 block type 或 block types。
- 当 extraction warnings 存在时，在 preview 附近显示。
- 继续支持 existing selected-source navigation commands。

Inspector 应保持 read-only 和 command-first。

## Trace and Ingest Summary Strategy

PDF ingest traces 和 summaries 应报告 counts 和 warnings。这让 PDF extraction 可审计，并帮助用户
区分 unsupported PDFs 与成功 ingest 但 chunks 很少的情况。

Proposed summary shape：

```json
{
  "pdf_files_seen": 2,
  "pdf_files_ingested": 2,
  "pdf_pages_seen": 18,
  "pdf_pages_with_text": 17,
  "pdf_tables_extracted": 5,
  "pdf_empty_pages": 1,
  "pdf_warnings": [
    {
      "source_path": "examples/knowledge/paper.pdf",
      "page": 4,
      "kind": "empty_page",
      "message": "No extractable text found on page."
    }
  ]
}
```

Exact schema 可在实现阶段细化，但 trace 应回答这些问题：

- 发现了多少 PDF files？
- ingest 了多少？
- 看到了多少 pages？
- 有多少 pages 有 extractable text？
- 抽取了多少 tables？
- 哪些 pages 产生 warnings？
- 使用了哪种 extraction method？

Existing trace workflows 应继续作为 `.ragent/` 下的 local JSON artifacts。

## Evaluation Dataset Considerations

PDF support 应让未来 retrieval eval cases 更精确，但不要求 v0.1.1 解决 answer evaluation。

Possible retrieval case shape：

```json
{
  "id": "case-pdf-001",
  "query": "What metrics are reported in the retrieval evaluation table?",
  "expected_source_paths": ["examples/knowledge/ragentforge_report.pdf"],
  "expected_pages": [7],
  "tags": ["pdf", "table", "retrieval-eval"]
}
```

Evaluation guidance：

- v0.1.1 仍可以只按 source path 评估。
- `expected_pages` 可以作为未来 optional field 添加。
- Table-aware cases 应在 PDF ingestion 落地后添加。
- Eval fixtures 应小且 licensing-safe。
- Retrieval eval 应保持 retrieval-only，不引入 LLM-as-judge behavior。

PDF-aware eval cases 可以测试 page 和 table metadata 是否经过 ingestion、chunking、retrieval
和 source display 后仍然保留。

## Library Choice

未来实现应评估 `pypdf`、`pdfplumber` 和 `PyMuPDF`。

### pypdf

`pypdf` 轻量，适合 basic text extraction 和 PDF metadata。但 table extraction 是 v0.1.1 的核心
需求，而 `pypdf` 本身不提供强 table extraction 或 layout-aware block modeling。

只有当实现需要 simple metadata inspection 或小 fallback path 时才使用 `pypdf`。它不应成为第一个
structured-ingestion choice。

### pdfplumber

`pdfplumber` 应成为 v0.1.1 首个 PDF structured ingestion library。它支持 page-level extraction、
layout-aware text extraction、coordinates 和 table extraction APIs。这些能力与 page-aware chunks、
table-aware source display、extraction warnings 和 TUI Inspector metadata 的目标一致。

Trade-off 是 PDF table extraction 仍可能需要 tuning，并且不是对所有文件都完美。只要 warnings 明确、
scope 限制在 text-based PDFs 内，这是可以接受的。

### PyMuPDF

`PyMuPDF` 对快速 PDF text/layout access 很强，未来可能对性能或更丰富 block extraction 有用。如果
pdfplumber 太慢，或额外 layout data 变得重要，可以之后再考虑。

### Recommendation

v0.1.1 首个 PDF structured ingestion library 使用 `pdfplumber`，因为 table extraction 和
layout-aware text extraction 很重要。在这个 design-only task 中不要添加依赖。

## Testing Strategy

未来实现应在 runtime code 改动时添加 tests。这个 design-only task 不应添加 runtime tests。

Future tests 应覆盖：

- 一份一到两页的 text-based PDF
- 包含简单 table 的 PDF
- 包含空白页或无 extractable text 页面的 PDF
- page、table 和 warning 的 ingest summary counts
- 带 `page_start` 和 `page_end` 的 chunk metadata
- 带 block type 和 table metadata 的 table chunks
- 带 page 和 table 信息的 source display labels
- TUI Inspector view-model 对 PDF metadata 的行为
- existing Markdown/TXT ingestion behavior
- unsupported 或 scanned-like PDF behavior，应明确且 warning-based

Test PDFs 应很小。它们应在 tests 中生成，或只在 licensing-safe 时 checked in。

Suggested test boundaries：

- 单元测试 table serialization，不依赖 PDF parsing。
- 单元测试 source label formatting，不调用 PDF extraction。
- 集成测试一个 tiny PDF 从 ingest 到 chunk storage。
- 除非测试专门关注 PDF-aware retrieval metadata，否则 semantic retrieval tests 应独立于 PDF
  extraction。

## Implementation Plan

### Step 1: PDF Design

当前 task。添加 `docs/PDF_INGESTION_DESIGN.md`，包含 planned scope、non-goals、
data model、source display、trace strategy、eval considerations、library recommendation 和
staged implementation plan。

### Step 2: PDF Page + Table Ingestion MVP

在这份设计文档创建后已经实现。实现保持 Markdown/TXT 使用 existing loader，并添加 additive
PDF path：`pdfplumber`、`DocumentBlock`、PDF-aware chunks、page/table metadata、source
labels、ingest summary warnings 和 focused tests。

- 添加选定的 PDF dependency。
- 在 ingest 中发现 `.pdf` files。
- 从 text-based PDFs 抽取 page text 和 page-local tables。
- 将 tables 序列化为 Markdown tables，带 TSV-like fallback。
- 创建 PDF-aware blocks 和 chunks。
- 将 page/table metadata 写入 chunk records。
- 更新 search 和 Ask 的 source display labels。
- 用 PDF counts 和 warnings 更新 ingest summary 和 trace fields。
- 为 text pages、simple tables、empty pages、metadata 和 Markdown/TXT compatibility 添加 tests。

### Step 3: PDF Reading Order and Formula Polish

- 改进 coordinate-aware block ordering。
- 减少明显 two-column interleaving。
- 保留 PDF text layer 中的 formula-like text。
- 为 suspicious extraction 或 uncertain ordering 添加 warnings。
- 改进 tables 和 formula-like blocks 周围的 chunking。

### Step 4: PDF Eval Seed Cases

- 添加一个小的 licensing-safe PDF knowledge fixture。
- 添加面向 PDF text 的 retrieval cases。
- 添加面向 PDF tables 的 retrieval cases。
- 在 path-only PDF eval 工作稳定后，再考虑 optional `expected_pages` support。

## Open Questions

- `DocumentBlock` 是否应在 PDF ingestion 的同时成为 Markdown/TXT 的 shared runtime model，
  还是先由 PDF 使用，之后再迁移现有 loaders？
- Table chunks 是否应总是 standalone，还是允许 table 前面的短 explanatory text 放在同一个
  chunk 中？
- 什么 warning threshold 应把 table 标记为 malformed？
- Page-aware eval 应在 v0.1.1 引入，还是推迟到 PDF retrieval fixtures 稳定之后？
- 未来 table serialization 是否应同时保存 Markdown 和 normalized cell arrays，以便更好检查？
