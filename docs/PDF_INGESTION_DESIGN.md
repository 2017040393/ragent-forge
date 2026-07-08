# PDF Structured Ingestion Design

> Language: English | [中文](PDF_INGESTION_DESIGN.zh-CN.md)

## Goal

Define the planned v0.1.1 PDF Structured Ingestion Foundation for RAGentForge.
This is a design document for future implementation work. It does not mean
RAGentForge v0.1.0 already supports PDF ingestion.

The goal is to support text-based PDFs in a way that stays local-first,
inspectable, source-grounded, traceable, eval-friendly, and compatible with
future retrieval quality work.

## Why PDF Support Matters

PDF support is a core product direction, not a small optional importer. Many
high-value knowledge sources are distributed as PDFs:

- papers
- technical reports
- textbooks
- manuals
- benchmark reports
- engineering documents

These sources often contain tables, page-level references, formula-like text,
captions, and dense multi-page arguments. Treating PDF ingestion as naive plain
text extraction would make search results harder to inspect and would hide
retrieval failures. The v0.1.1 design should preserve enough structure for
sources, traces, the TUI Inspector, and future retrieval evaluation to explain
where a result came from.

## Version Scope

v0.1.1 should design support for:

- text-based PDF ingestion
- page-aware metadata
- table extraction
- Markdown table serialization, with TSV-like text as a fallback
- page/table-aware source display
- PDF block metadata
- PDF extraction warnings
- traceable PDF ingest summaries
- TUI Inspector display of PDF page/table metadata
- future retrieval eval cases that can reference PDF source paths and
  optionally pages

Runtime PDF ingestion is not implemented by this document. The future v0.1.1
implementation should add code, dependencies, and tests in small steps after
this design is accepted.

Implementation note for the current implementation: the v0.1.1 PDF Page +
Table Ingestion MVP was implemented after this design was written. Older
v0.1.0 documentation may still describe PDF support as future work, but the
current implementation includes text-based PDF page/table ingestion.

## v0.1.2 PDF Extraction Quality Polish

The current implementation uses v0.1.1 as the page/table ingestion MVP and
extends it with a v0.1.2 quality-polish milestone for text-based PDFs. v0.1.1
answers the first runtime question: can RAGentForge discover PDFs, extract
page text and page-local tables, produce page/table-aware chunks, and show
PDF source labels such as `paper.pdf p.7 table 2`? v0.1.2 keeps that behavior
and improves the quality of extracted content before retrieval sees it.

v0.1.2 does not add OCR, scanned-PDF support, image text recognition, PDF
viewing, PDF opening, PDF editing, or full layout reconstruction. It remains
local-only and deterministic, using the text layer exposed by `pdfplumber`.
If a page has no extractable text, the extractor should keep reporting that as
a warning instead of pretending scanned content was processed.

The v0.1.2 scope is:

- Reading order baseline: prefer coordinate/word-aware ordering when
  `pdfplumber` exposes words and coordinates, preserve page boundaries, avoid
  obvious two-column left/right interleaving, and record
  `reading_order_strategy` plus fallback diagnostics.
- Table caption/context enhancement: detect nearby same-page captions such as
  `Table 2: ...`, `TABLE II ...`, or `表 1 ...`, prepend safe captions to
  table chunk text, and store caption/context metadata without merging
  separate tables.
- Conservative table/text de-duplication: remove only high-confidence table
  row text duplicated in paragraph text, never remove table blocks, and record
  how many lines/pages were affected.
- Formula-like text preservation: keep formula-like text from the PDF text
  layer in paragraph blocks and tag related blocks/chunks with
  `possible_formula` and bounded `possible_formula_lines` metadata.
- Header/footer filtering and diagnostics: conservatively filter repeated
  short header/footer lines across pages, preserve body text and page metadata,
  and record counts for suspected headers/footers filtered.
- Diagnostics propagation: carry the new quality metadata through ingest
  summary, ingest trace metadata, chunk metadata, search results, and concise
  TUI Inspector/source displays.

These improvements are intentionally conservative. When confidence is low, the
extractor should prefer keeping text and surfacing diagnostics over deleting
potentially useful content.

## Non-Goals

v0.1.1 PDF structured ingestion should explicitly exclude:

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

Existing Markdown/TXT ingestion, search, Ask, trace, eval, and TUI behavior
must keep working. PDF metadata should be additive and backward-compatible.

## Supported PDF Type

Only text-based PDFs are supported in v0.1.1. Scanned PDFs and OCR are not
supported.

Text-based PDFs are PDFs where text can be selected, copied, and extracted
from a text layer. A PDF may still have individual pages with no extractable
text. Those pages should produce warnings instead of silent failures.

Unsupported or scanned-like PDFs should not be treated as successful empty
documents. The ingest summary and trace should explain that no extractable text
was found, and future CLI output should surface that warning clearly.

## Extraction Requirements

### Page-Aware Text Extraction

PDF ingestion should process each file page by page. Each extracted text block
must retain the source path and page number. Page numbers shown to users should
be one-based because that matches how readers cite PDFs.

At minimum, the future extractor should record:

- `source_path`
- `media_type: "application/pdf"`
- one-based `page_number`
- page-local block order
- extracted text
- extraction method
- warnings for empty or suspicious pages

Page breaks should be preserved in metadata even when nearby blocks are later
combined into larger chunks.

### Table Extraction

Table extraction is a core requirement for v0.1.1. Tables must not be silently
flattened into unreadable paragraph text.

The future extractor should detect tables using the chosen PDF library's table
extraction API, normalize rows and cells, and serialize tables as Markdown when
possible:

```markdown
| Method | Hit@1 | MRR |
|---|---:|---:|
| lexical | 0.67 | 0.78 |
| hybrid | 1.00 | 1.00 |
```

If Markdown table serialization is unreliable for a specific table, the
fallback should be TSV-like text. The fallback should still retain table
metadata so source displays and evaluation can identify the result as a table.

Each extracted table should retain:

- one-based page number
- page-local table index
- block type `table`
- row count
- column count
- serialization format, such as `markdown_table` or `tsv`
- extraction warnings, when applicable

Table extraction should warn when a detected table is empty, has no usable
cells, has inconsistent row widths after normalization, or produces a serialized
form that is too short to be useful.

### Reading Order

v0.1.1 should improve obvious reading-order problems without promising perfect
layout reconstruction.

The future extractor should:

- extract at page level first
- preserve block-level order when the library provides reliable layout data
- use coordinate-aware sorting when block coordinates are available
- avoid obvious two-column interleaving when possible
- preserve page breaks in metadata
- reserve header/footer detection as a future improvement

The initial implementation can prefer simple, auditable ordering rules over
opaque layout magic. If ordering is uncertain, the extractor should record a
warning rather than pretending the page was perfectly reconstructed.

### Formula Text Preservation

Formula handling is important but secondary to text and table extraction.

The future extractor should preserve formula-like text when it exists in the
PDF text layer. It should not intentionally delete mathematical symbols, Greek
letters, operators, subscripts, superscripts, or equation labels when the text
layer exposes them.

v0.1.1 should not:

- perform OCR for formulas embedded as images
- perform image formula recognition
- attempt full LaTeX reconstruction
- guarantee mathematically perfect formula layout

A future block metadata flag such as `possible_formula: true` can help search,
sources, and the Inspector identify blocks that may need manual inspection.

### Metadata

PDF extraction should produce structured metadata early, before chunking. This
keeps PDF-specific behavior concentrated in ingestion rather than scattered
through retrieval, Ask, eval, and TUI code.

Expected metadata includes:

- media type
- source path
- page number or page range
- block index
- block type
- table indices
- section title, when recoverable
- extraction method
- extraction warnings
- optional coordinate/layout metadata

### Warnings and Failure Modes

PDF extraction problems should be inspectable, not silent.

Warnings should be structured enough to appear in ingest summaries, traces,
and future CLI/TUI inspection surfaces. Warning kinds may include:

- `empty_page`
- `no_extractable_text`
- `table_empty`
- `table_malformed`
- `reading_order_uncertain`
- `unsupported_scanned_pdf`
- `formula_text_suspicious`

A warning should include at least source path, page number when available,
kind, and a human-readable message.

## Proposed PDF Block Model

RAGentForge should introduce a middle-layer document block model before chunks:

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

This class is conceptual in this design. It should not be implemented until the
runtime PDF ingestion work begins.

The intended pipeline is:

```text
PDF / Markdown / TXT
-> DocumentBlock[]
-> Chunk[]
-> SearchResult
-> Ask sources
-> TUI Inspector
```

The block layer gives each document format one place to express structure. PDF
can emit page-aware paragraph and table blocks. Markdown can later emit heading
and list blocks. TXT can emit paragraph or unknown blocks. Chunking and source
display can then consume a shared shape instead of branching on PDF details in
multiple services.

Example PDF table block metadata:

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

PDF chunks should be created from ordered `DocumentBlock` records. The chunker
should remain deterministic and should retain source metadata from all blocks
that contributed to a chunk.

Expected PDF chunk metadata:

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

Chunking rules:

- Paragraph blocks may be combined up to the existing chunk size target.
- Table blocks should usually become standalone chunks or anchor the chunk that
  immediately describes them.
- A table chunk must retain `block_type: table` or include `table` in
  `block_types`.
- Chunks spanning pages must set `page_start` and `page_end`.
- Chunks with extraction warnings should carry them forward so retrieval
  results and traces can explain suspicious content.
- Existing Markdown/TXT chunk metadata must remain valid.

The first implementation should avoid complex cross-page table reconstruction.
If a table appears to continue across pages, each page-local table can be
extracted separately with a warning such as `cross_page_table_possible`.

## Source Display Strategy

Search, Ask, and TUI source displays should use additive PDF labels while
leaving Markdown/TXT source labels unchanged.

Recommended labels:

```text
paper.pdf p.7
paper.pdf pp.7-8
paper.pdf p.7 table 2
```

Source label rules:

- Use the basename for compact display and retain the full `source_path` in
  metadata.
- Use `p.N` for one page.
- Use `pp.N-M` for a page range.
- Append `table K` when the primary chunk content is table `K` or when the
  chunk has a single table index.
- Do not require local file opening or PDF viewer integration.
- Keep Markdown/TXT behavior unchanged when page metadata is absent.

Ask sources should include the same PDF page/table labels used by search so a
user can trace an answer back to the page-level source.

## TUI Inspector Strategy

The TUI Inspector should display PDF source metadata as structured text. It
should not become a PDF viewer, and it should not open local files.

Example Inspector display:

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

Inspector behavior:

- Show `type: pdf` when `media_type` is `application/pdf`.
- Show page range when `page_start` and `page_end` exist.
- Show table index when table metadata exists.
- Show block type or block types when available.
- Show extraction warnings near the preview when present.
- Continue to support existing selected-source navigation commands.

The Inspector should remain read-only and command-first.

## Trace and Ingest Summary Strategy

PDF ingest traces and summaries should report counts and warnings. This keeps
PDF extraction auditable and helps users distinguish unsupported PDFs from
successful ingests with few chunks.

Proposed summary shape:

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

The exact schema can be refined during implementation, but the trace should
answer these questions:

- How many PDF files were discovered?
- How many were ingested?
- How many pages were seen?
- How many pages had extractable text?
- How many tables were extracted?
- Which pages produced warnings?
- Which extraction method was used?

Existing trace workflows should stay local JSON artifacts under `.ragent/`.

## Evaluation Dataset Considerations

PDF support should make future retrieval eval cases more precise without
requiring v0.1.1 to solve answer evaluation.

Possible retrieval case shape:

```json
{
  "id": "case-pdf-001",
  "query": "What metrics are reported in the retrieval evaluation table?",
  "expected_source_paths": ["examples/knowledge/ragentforge_report.pdf"],
  "expected_pages": [7],
  "tags": ["pdf", "table", "retrieval-eval"]
}
```

Evaluation guidance:

- v0.1.1 may still evaluate by source path only.
- `expected_pages` can be added as a future optional field.
- Table-aware cases should be added after PDF ingestion lands.
- Eval fixtures should be small and licensing-safe.
- Retrieval eval should remain retrieval-only and should not introduce
  LLM-as-judge behavior.

PDF-aware eval cases will help test whether page and table metadata survive
ingestion, chunking, retrieval, and source display.

## Library Choice

The future implementation should evaluate `pypdf`, `pdfplumber`, and
`PyMuPDF`.

### pypdf

`pypdf` is lightweight and useful for basic text extraction and PDF metadata.
However, table extraction is a core v0.1.1 requirement, and `pypdf` does not
provide strong table extraction or layout-aware block modeling by itself.

Use `pypdf` only if the implementation needs simple metadata inspection or a
small fallback path. It should not be the first structured-ingestion choice.

### pdfplumber

`pdfplumber` should be the first PDF structured ingestion library for v0.1.1.
It supports page-level extraction, layout-aware text extraction, coordinates,
and table extraction APIs. Those capabilities align with the goals of
page-aware chunks, table-aware source display, extraction warnings, and TUI
Inspector metadata.

The trade-off is that PDF table extraction can still require tuning and is not
perfect across all files. That is acceptable if warnings are explicit and the
scope stays limited to text-based PDFs.

### PyMuPDF

`PyMuPDF` is strong for fast PDF text/layout access and may be useful for
future performance or richer block extraction work. It can be considered later
if pdfplumber is too slow or if additional layout data becomes important.

### Recommendation

Use `pdfplumber` as the first PDF structured ingestion library for v0.1.1
because table extraction and layout-aware text extraction matter. Do not add
the dependency in this design-only task.

## Testing Strategy

The future implementation should add tests when runtime code changes. This
design-only task should not add runtime tests.

Future tests should cover:

- a text-based PDF with one or two pages
- a PDF with a simple table
- a PDF with an empty page or page without extractable text
- ingest summary counts for pages, tables, and warnings
- chunk metadata with `page_start` and `page_end`
- table chunks with block type and table metadata
- source display labels with page and table information
- TUI Inspector view-model behavior for PDF metadata
- existing Markdown/TXT ingestion behavior
- unsupported or scanned-like PDF behavior that is explicit and warning-based

Test PDFs should be small. They should be generated during tests or checked in
only when licensing is safe.

Suggested test boundaries:

- Unit-test table serialization separately from PDF parsing.
- Unit-test source label formatting without invoking PDF extraction.
- Integration-test one tiny PDF through ingest and chunk storage.
- Keep semantic retrieval tests independent from PDF extraction unless the
  test is specifically about PDF-aware retrieval metadata.

## Implementation Plan

### Step 1: PDF Design

Current task. Add `docs/PDF_INGESTION_DESIGN.md` with the planned scope,
non-goals, data model, source display, trace strategy, eval considerations,
library recommendation, and staged implementation plan.

### Step 2: PDF Page + Table Ingestion MVP

Implemented after this design document was created. The implementation keeps
Markdown/TXT on the existing loader and adds an additive PDF path with
`pdfplumber`, `DocumentBlock`, PDF-aware chunks, page/table metadata, source
labels, ingest summary warnings, and focused tests.

- Add the selected PDF dependency.
- Discover `.pdf` files during ingest.
- Extract page text and page-local tables from text-based PDFs.
- Serialize tables as Markdown tables, with TSV-like fallback.
- Create PDF-aware blocks and chunks.
- Write page/table metadata into chunk records.
- Update source display labels for search and Ask.
- Update ingest summary and trace fields with PDF counts and warnings.
- Add tests for text pages, simple tables, empty pages, metadata, and
  Markdown/TXT compatibility.

### Step 3: PDF Reading Order and Formula Polish

- Improve coordinate-aware block ordering.
- Reduce obvious two-column interleaving.
- Preserve formula-like text from the PDF text layer.
- Add warnings for suspicious extraction or uncertain ordering.
- Improve chunking around tables and formula-like blocks.

### Step 4: PDF Eval Seed Cases

- Add a small licensing-safe PDF knowledge fixture.
- Add retrieval cases targeting PDF text.
- Add retrieval cases targeting PDF tables.
- Consider optional `expected_pages` support after path-only PDF eval works.

## Open Questions

- Should `DocumentBlock` become a shared runtime model for Markdown/TXT at the
  same time as PDF ingestion, or should PDF use it first and migrate existing
  loaders later?
- Should table chunks always be standalone, or should short explanatory text
  before a table be allowed in the same chunk?
- What warning threshold should mark a table as malformed?
- Should page-aware eval be introduced in v0.1.1 or deferred until PDF
  retrieval fixtures are stable?
- Should future table serialization store both Markdown and normalized cell
  arrays for better inspection?
