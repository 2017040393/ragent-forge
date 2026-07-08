# RAGentForge v0.1-alpha-1 Structured Ingestion Notes

> 语言: [English](RELEASE_NOTES_V0_1_ALPHA_1.md) | 中文

这些 notes 描述 PDF 和 structured ingestion 工作之后的 v0.1-alpha-1 structured
ingestion milestone。本文保留 milestone notes，但不创建或声称一个单独发布的 tag。

## Summary

Structured ingestion milestone 把 RAGentForge 从 Markdown/TXT-only plain text
ingestion 扩展为 Markdown、TXT 和 PDF 共享的 structured ingestion foundation。

核心 pipeline 现在是：

```text
local file
-> structured loader
-> Document + DocumentBlock[]
-> BlockChunker
-> DocumentChunk[]
-> retrieval / ask / traces / TUI inspection
```

这样 retrieval 仍保持 chunk-based，同时 ingestion 获得一种共同中间表示，用来承载
format-specific metadata。

## Highlights

- 在 structured ingestion milestone 中加入 PDF page 和 table ingestion。
- 加入 PDF extraction quality polish：reading order、重复 header/footer filtering、
  table text de-duplication，以及 possible formula metadata。
- 为 Markdown、TXT 和 PDF 加入统一 structured loader layer。
- 让 `DocumentBlock` 成为 supported local file types 的共同中间表示。
- 泛化 `BlockChunker`，使其不再 hard-code PDF metadata。
- 加入 lightweight Markdown block detection：headings、paragraphs、pipe tables、
  lists、fenced code blocks 和 blockquotes。
- 加入按空行分割的 TXT paragraph blocks。
- 通过 chunk metadata 保留 Markdown/TXT character ranges。
- 保留 PDF page、table、reading order、quality 和 warning metadata。
- 在 inspectors 中加入 Markdown/TXT source metadata display，同时保持 compact source
  labels 稳定。

## Included Since v0.1.0

### v0.1.1 PDF Page + Table Ingestion MVP

- `ragent ingest` 除 Markdown/TXT 外也接受 `.pdf` 文件。
- PDF text 会作为 page-aware structured blocks 加载。
- PDF tables 会序列化为可切块的 table text。
- PDF chunks 可以显示 `p.7` 或 `pp.7-8` 这样的 page ranges。
- PDF table chunks 可以显示 `paper.pdf p.7 table 2` 这样的 labels。

### v0.1.2 PDF Extraction Quality Polish

- 改进 PDF reading order handling。
- 加入 repeated header/footer filtering metadata。
- 加入 table text de-duplication metadata。
- 为 formula-like lines 加入 possible formula metadata。
- 将 extraction warnings 传播到 chunk metadata 和 inspectors。

### v0.1-alpha-1 Unified Structured Ingestion Foundation

- Markdown、TXT 和 PDF 现在都会在 chunking 前流经 `DocumentBlock[]`。
- `IngestService` 对每种 supported format 都使用 structured loader layer。
- `BlockChunker` 从 blocks 聚合 common metadata，并且只对 PDF chunks 应用 PDF-only
  metadata。
- Markdown headings 会把 `section_title` 和 `heading_path` metadata 附加到后续内容。
- Markdown pipe tables 默认保持 standalone chunks。
- TXT paragraphs 变成带 `text/plain` metadata 的 paragraph blocks。

## Compatibility

Public command surface 保持不变：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
uv run ragent chunks list --workspace .ragent
uv run ragent chunks show "<chunk_id>" --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
uv run ragent tui --workspace .ragent
```

Existing retrieval modes、workspace paths、generation behavior、embedding
configuration 和 TUI command names 保持不变。

Markdown/TXT chunk boundaries 可能与 v0.1.0 不同，因为它们现在会先遵循 structured
blocks 再进行 chunk splitting。Markdown/TXT chunks 仍保留 character ranges。

## Demo

使用 structured ingestion demo workflow：

- [STRUCTURED_INGESTION_DEMO.zh-CN.md](STRUCTURED_INGESTION_DEMO.zh-CN.md)

该 demo 展示如何 ingest 一个混合 Markdown/TXT/PDF corpus、检查 chunk metadata、验证
Markdown sections 和 TXT ranges，并搜索生成的 chunks。

## Known Limitations

- Markdown parsing 有意保持 lightweight 和 line-based。
- Markdown loader 不是完整 CommonMark parser。
- Markdown table detection 仅限简单 pipe tables。
- TXT ingestion 只按空行分割 paragraphs。
- 不包含 OCR 和 scanned PDF support。
- 不包含 PDF viewing、opening、editing 和 source full-text viewing。
- Reranking 或 query expansion 等 retrieval quality improvements 不在这个 milestone
  范围内。
- TUI 对 ingest/index/eval/config workflows 保持 read-only。

## Verification Checklist

把这些 milestone notes 转成 release material 前，运行：

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

然后运行 structured ingestion demo：

```text
docs/STRUCTURED_INGESTION_DEMO.zh-CN.md
```

如果创建实际 GitHub release，请更新 tag names、screenshot links 和 release status。
