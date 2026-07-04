# RAGentForge v0.1 Demo Script

> Language: English | [中文](DEMO_SCRIPT.zh-CN.md)

## Demo Goal

Show RAGentForge as a local-first, inspectable RAG console. In 3-5 minutes, the
demo should prove that local Markdown files can be ingested, chunked, searched,
asked over with sources, traced, evaluated, and inspected through the
command-first TUI Shell.

The demo should stay honest: semantic and hybrid retrieval require a configured
embedding provider and a built vector index, generation may be disabled with the
default `null` provider, and Shell Ask does not write new traces in v0.1.

For the `Develop_PDF` structured ingestion branch, use
[STRUCTURED_INGESTION_DEMO.md](STRUCTURED_INGESTION_DEMO.md) after this base
demo. That branch workflow shows Markdown, TXT, and PDF flowing through
`DocumentBlock[] -> BlockChunker -> DocumentChunk[]`.

## 30-Second Project Introduction

RAGentForge is a local-first RAG console for inspecting the full retrieval
workflow. It stores derived state in `.ragent`, supports deterministic chunking,
lexical, semantic, and hybrid retrieval, optional source-grounded generation,
operation traces, retrieval evaluation, and a command-first Textual TUI.

The core idea is inspectability. Instead of hiding the RAG pipeline behind a
hosted app or framework abstraction, the project exposes chunks, sources,
traces, eval reports, and TUI transcript state as local artifacts.

## Demo Setup

Run commands from the repository root.

Install dependencies if needed:

```bash
uv sync --extra dev
```

Start from a clean demo workspace:

```bash
rm -rf .ragent
```

On Windows PowerShell, use:

```powershell
Remove-Item -Recurse -Force .ragent
```

## Demo Flow

### 1. Clean Workspace

Explain that `.ragent` is derived local state and can be regenerated.

```bash
rm -rf .ragent
uv run ragent status --workspace .ragent
```

Expected story: the source documents are still in `examples/knowledge`, but the
derived workspace has not been prepared yet.

### 2. Ingest Local Knowledge

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
```

Point out that ingestion writes chunks, an ingest summary, and a CLI operation
trace under `.ragent`.

### 3. Inspect Chunks

```bash
uv run ragent chunks list --workspace .ragent
```

Optional, if time allows:

```bash
uv run ragent chunks show "<chunk_id>" --workspace .ragent
```

Say that deterministic chunk ids and JSONL storage make the pipeline easier to
debug and test.

### 4. Run Lexical Search

```bash
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

Explain that lexical search works immediately after ingestion and is useful for
exact terms, file names, configuration fields, and quick local demos.

### 5. Ask with Sources

```bash
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

If generation is still using the default `null` provider, say that this is
expected: Ask stays in retrieval-only mode and shows retrieved context with
sources. If `openai_responses` is configured, this command can generate a
source-grounded answer.

### 6. Inspect Trace

```bash
uv run ragent traces latest --workspace .ragent
```

Explain that CLI workflows write traces. This is important because it makes the
system inspectable after each operation. Shell Ask does not write new traces in
v0.1; CLI `ragent ask` is the trace-producing Ask workflow.

### 7. Run Retrieval Evaluation

```bash
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
```

Point out hit@k and MRR. Say that this is retrieval-only evaluation, not answer
quality evaluation or LLM-as-judge.

### 8. Launch the TUI Shell

```bash
uv run ragent tui
```

Mention that the TUI reads the default `.ragent` workspace from the current
working directory. It does not accept a `--workspace` argument in v0.1.

### 9. Search in the TUI

Type into the composer:

```text
/help
/search Agentic RAG
```

Point out inline command suggestions, background Shell Search, source lists in
the transcript, and the selected-source Inspector.

### 10. Navigate Sources

Type:

```text
/source 2
/sources
/source next
/source prev
```

Explain that source navigation is command-first. There is no source table UI or
mouse selection in v0.1.

### 11. Show Read-Only Workspace State

Type:

```text
What is Agentic RAG?
/trace
/settings
/exit
```

Explain that ordinary text runs Shell Ask in a background worker. `/trace` and
`/settings` are read-only inspection commands. Use `/exit`, `/quit`, or `/q`
from the composer to quit; there are no global single-key shortcuts such as `q`.

## Reference Screenshots

Use these assets when presenting v0.1 in a portfolio, README, or GitHub Release:

![TUI Shell search](assets/v0_1/tui-shell-search.jpg)

![TUI source inspector](assets/v0_1/tui-source-inspector.jpg)

![TUI trace and settings](assets/v0_1/tui-trace-settings.jpg)

![Retrieval evaluation output](assets/v0_1/tui-retrieval-eval.jpg)

## What to Say During the Demo

- "The source documents stay local; `.ragent` contains derived artifacts."
- "Chunking is deterministic so failures are easier to reproduce."
- "Retrieval modes are explicit: lexical, semantic, and hybrid."
- "Semantic and hybrid retrieval require embeddings and a built vector index."
- "Ask works without generation. With the default `null` provider, it stays in
  retrieval-only mode."
- "CLI workflows write traces. The TUI reads and displays those traces."
- "The TUI is intentionally command-first rather than dashboard-first."
- "This is not an autonomous agent framework. It is a small inspectable RAG MVP."

## What This Demonstrates Technically

- Local workspace design with plain files under `.ragent`.
- Markdown/TXT ingestion and deterministic chunking.
- Lexical retrieval without embeddings.
- Semantic and hybrid retrieval as optional indexed modes.
- Source-grounded Ask and retrieval-only fallback behavior.
- CLI operation traces for debugging and explanation.
- Retrieval evaluation with hit@k and MRR over JSONL cases.
- Shared application services used by CLI and TUI.
- Textual Shell workers for non-blocking Ask/Search.
- Command suggestions, source navigation, and selected-source inspection in the
  TUI.

For the structured ingestion branch, also demonstrate:

- PDF page/table ingestion.
- Markdown/TXT/PDF as structured `DocumentBlock` records before chunking.
- Markdown section metadata such as `section_title` and `heading_path`.
- TXT character ranges preserved through `BlockChunker`.
- PDF page/table metadata preserved without adding PDF-only fields to
  Markdown/TXT chunks.

## Fallback Path If Embeddings Are Not Configured

Keep the demo lexical-only:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
uv run ragent tui
```

In the TUI, use:

```text
/mode lexical
/search Agentic RAG
What is Agentic RAG?
/sources
/source next
/trace
/exit
```

Say explicitly that semantic and hybrid modes are supported but require an
embedding provider and `uv run ragent index build --workspace .ragent`.

## Closing Summary

RAGentForge v0.1 demonstrates a complete local RAG loop: ingest, chunk, search,
ask, trace, evaluate, and inspect through a command-first TUI. It deliberately
avoids production claims and autonomous agent behavior. The value of the project
is that it makes the RAG pipeline visible, reproducible, and easy to discuss in
an interview or portfolio review.
