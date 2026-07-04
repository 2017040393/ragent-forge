# RAGentForge v0.1.3 Structured Ingestion Branch Notes

These notes describe the `Develop_PDF` branch after the PDF and structured
ingestion work. They are suitable as draft release notes when the branch is
merged and tagged, but this document does not create or claim a released tag.

## Summary

The structured ingestion branch extends RAGentForge from Markdown/TXT-only
plain text ingestion to a shared structured ingestion foundation for
Markdown, TXT, and PDF.

The core pipeline is now:

```text
local file
-> structured loader
-> Document + DocumentBlock[]
-> BlockChunker
-> DocumentChunk[]
-> retrieval / ask / traces / TUI inspection
```

This keeps retrieval chunk-based while giving ingestion a common intermediate
representation for format-specific metadata.

## Highlights

- Added PDF page and table ingestion on the `Develop_PDF` branch.
- Added PDF extraction quality polish for reading order, repeated
  header/footer filtering, table text de-duplication, and possible formula
  metadata.
- Added a unified structured loader layer for Markdown, TXT, and PDF.
- Made `DocumentBlock` the common intermediate representation for supported
  local file types.
- Generalized `BlockChunker` so it no longer hard-codes PDF metadata.
- Added lightweight Markdown block detection for headings, paragraphs, pipe
  tables, lists, fenced code blocks, and blockquotes.
- Added TXT paragraph blocks split by blank lines.
- Preserved Markdown/TXT character ranges through chunk metadata.
- Preserved PDF page, table, reading order, quality, and warning metadata.
- Added Markdown/TXT source metadata display in inspectors while keeping compact
  source labels stable.

## Included Since v0.1.0

### v0.1.1 PDF Page + Table Ingestion MVP

- `ragent ingest` accepts `.pdf` files in addition to Markdown/TXT.
- PDF text is loaded as structured page-aware blocks.
- PDF tables are serialized into chunkable table text.
- PDF chunks can show page ranges such as `p.7` or `pp.7-8`.
- PDF table chunks can show labels such as `paper.pdf p.7 table 2`.

### v0.1.2 PDF Extraction Quality Polish

- Improved PDF reading order handling.
- Added repeated header/footer filtering metadata.
- Added table text de-duplication metadata.
- Added possible formula metadata for formula-like lines.
- Added extraction warning propagation into chunk metadata and inspectors.

### v0.1.3 Unified Structured Ingestion Foundation

- Markdown, TXT, and PDF now all flow through `DocumentBlock[]` before chunking.
- `IngestService` uses the structured loader layer for every supported format.
- `BlockChunker` aggregates common metadata from blocks and applies PDF-only
  metadata only to PDF chunks.
- Markdown headings attach `section_title` and `heading_path` metadata to
  following content.
- Markdown pipe tables remain standalone chunks by default.
- TXT paragraphs become paragraph blocks with `text/plain` metadata.

## Compatibility

The public command surface is unchanged:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
uv run ragent chunks list --workspace .ragent
uv run ragent chunks show "<chunk_id>" --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
uv run ragent tui
```

Existing retrieval modes, workspace paths, generation behavior, embedding
configuration, and TUI command names are unchanged.

Markdown/TXT chunk boundaries may differ from v0.1.0 because they now follow
structured blocks before chunk splitting. Character ranges remain available for
Markdown/TXT chunks.

## Demo

Use the structured ingestion demo workflow:

- [STRUCTURED_INGESTION_DEMO.md](STRUCTURED_INGESTION_DEMO.md)

The demo shows how to ingest a mixed Markdown/TXT/PDF corpus, inspect chunk
metadata, verify Markdown sections and TXT ranges, and search over the generated
chunks.

## Known Limitations

- Markdown parsing is intentionally lightweight and line-based.
- The Markdown loader is not a full CommonMark parser.
- Markdown table detection is limited to simple pipe tables.
- TXT ingestion only splits paragraphs by blank lines.
- OCR and scanned PDF support are not included.
- PDF viewing, opening, editing, and source full-text viewing are not included.
- Retrieval quality improvements such as BM25, reranking, or query expansion
  are outside this branch.
- The TUI remains read-only for ingest/index/eval/config workflows.

## Verification Checklist

Before turning these branch notes into a tagged release, run:

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

Then run the structured ingestion demo from:

```text
docs/STRUCTURED_INGESTION_DEMO.md
```

If creating an actual GitHub release, update tag names, screenshot links, and
merge status after the branch lands.
