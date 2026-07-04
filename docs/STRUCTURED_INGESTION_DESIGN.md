# v0.1.3 Unified Structured Ingestion Foundation

This branch adds a unified structured ingestion foundation for local knowledge
files. It is a branch-level design note and does not claim a released version
until the branch is merged and released.

Related branch documents:

- [RELEASE_NOTES_V0_1_3.md](RELEASE_NOTES_V0_1_3.md)
- [STRUCTURED_INGESTION_DEMO.md](STRUCTURED_INGESTION_DEMO.md)

## Why This Exists

PDF ingestion introduced a structured path because page text, tables, reading
order, formulas, and extraction warnings need more context than plain character
windows can carry. Markdown and TXT previously still used plain text chunking,
which left the ingestion architecture split across two models.

The v0.1.3 foundation makes `DocumentBlock` the common intermediate
representation for supported local formats:

```text
Markdown -> Document + DocumentBlock[] -> BlockChunker -> DocumentChunk[]
TXT      -> Document + DocumentBlock[] -> BlockChunker -> DocumentChunk[]
PDF      -> Document + DocumentBlock[] -> BlockChunker -> DocumentChunk[]
```

Retrieval remains chunk-based after ingestion. Search, ask, indexing, and trace
flows continue to consume `DocumentChunk` records without needing to know which
loader produced them.

## Loader Layer

`core.ingestion.structured_loader` is the format dispatch layer. It owns the
supported extension registry and routes each file to a structured loader that
returns a `StructuredLoadResult`:

- `document`: the source document payload and document-level metadata.
- `blocks`: ordered `DocumentBlock` records.
- `metadata`: loader-level summary metadata.
- `warnings`: optional loader warnings, used by PDF today.

This keeps `IngestService` format-agnostic. Adding another local document type
should mean adding a loader and registering its extension, not adding another
chunking branch to the service.

## Markdown Blocks

Markdown uses deterministic, line-based parsing rather than a full Markdown AST.
The goal is useful retrieval structure, not CommonMark compliance.

Supported lightweight block types:

- `heading`: ATX headings from `#` through `######`.
- `paragraph`: consecutive non-blank lines that are not another supported block.
- `table`: simple pipe tables with a separator row.
- `list`: consecutive unordered or ordered list lines.
- `code`: fenced code blocks using backtick or tilde fences.
- `blockquote`: consecutive quote lines beginning with `>`.

Markdown block metadata includes:

- `media_type: text/markdown`
- `start_char` and `end_char`
- `section_title`
- `heading_path`
- block-specific fields such as `heading_level`, `serialization`, or
  `code_language` when available

Markdown tables remain standalone chunks by default so table text is not merged
into surrounding paragraphs.

## TXT Blocks

TXT ingestion creates paragraph blocks by splitting on blank lines. Each block
uses:

- `media_type: text/plain`
- `block_type: paragraph`
- `start_char` and `end_char`

Long TXT paragraphs can remain one block. `BlockChunker` handles oversized
non-table blocks with the configured chunk size and overlap.

## PDF Blocks

PDF keeps the structured behavior added by the PDF ingestion milestones.
PDF chunks continue to preserve:

- `media_type: application/pdf`
- `page_start` and `page_end`
- `table_indices`
- reading order metadata
- table captions and table context
- possible formula metadata
- header/footer filtering metadata
- extraction warnings

PDF-only fields are not added to Markdown or TXT chunks.

## Chunk Metadata

`BlockChunker` now builds common metadata from the blocks it receives:

- `source_path`
- `media_type`
- `block_types`
- `block_type` when a chunk has exactly one block type

For Markdown and TXT, character ranges are aggregated across combined blocks.
For split oversized text blocks, each chunk gets the character range for its
window. For Markdown, the first meaningful `section_title` and `heading_path`
are retained.

For PDF, existing page/table/quality metadata is added only when the chunk media
type is `application/pdf`.

## Display Behavior

Compact source labels remain stable:

- Markdown: `rag.md`
- TXT: `notes.txt`
- PDF: `paper.pdf p.7 table 2`

Inspector views can show structured source metadata:

- Markdown: type, section, heading path, block type.
- TXT: type and block type.
- PDF: existing page, table, reading order, quality, and warning details.
