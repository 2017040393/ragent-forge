# RAGentForge v0.1.0 Release Notes

> Language: English | [中文](RELEASE_NOTES_V0_1.zh-CN.md)

## Summary

RAGentForge v0.1.0 is a local-first, inspectable RAG MVP for Markdown/TXT
knowledge bases. It provides a full local loop for ingestion, deterministic
chunking, lexical/semantic/hybrid retrieval, source-grounded Ask, CLI operation
traces, retrieval evaluation, and command-first TUI inspection.

This file can be copied into a GitHub Release for tag `v0.1.0`.

## Highlights

- Local `.ragent` workspace with inspectable derived artifacts.
- Markdown/TXT ingestion and deterministic chunking.
- Lexical retrieval without embeddings.
- Semantic retrieval through an OpenAI-compatible embedding provider and local
  JSONL vector index.
- Hybrid RRF retrieval over lexical and semantic candidates.
- Ask pipeline with optional OpenAI Responses-compatible generation.
- Retrieval-only Ask mode with the default `null` generation provider.
- Answer and source display for CLI and TUI workflows.
- CLI operation traces for ingest, index build, search, ask, and retrieval eval.
- Retrieval evaluation with hit@k and MRR over JSONL cases.
- Command-first Textual TUI Shell with inline command suggestions, background
  Ask/Search workers, source navigation, and selected-source Inspector.

## Included in v0.1.0

- `ragent ingest` for local Markdown/TXT ingestion.
- `ragent status` for workspace status.
- `ragent config show` and `ragent config init` for local provider config.
- `ragent chunks list` and `ragent chunks show` for chunk inspection.
- `ragent index build` and `ragent index status` for semantic index workflows.
- `ragent search` with `lexical`, `semantic`, and `hybrid` retrieval modes.
- `ragent ask` with retrieval-only and optional generated-answer modes.
- `ragent traces latest`, `ragent traces list`, and `ragent traces show`.
- `ragent eval retrieval` with hit@1, hit@3, hit@5, requested hit@k, MRR, and
  failed-case reporting.
- `ragent tui` for the command-first Textual Shell.

## Command-First TUI

The TUI is a single Shell interface with a transcript, composer, status line,
inline command suggestions, and Inspector panel.

Current Shell commands include:

```text
/help
/mode lexical|semantic|hybrid
/limit <n>
/context <n>
/prompt on|off
/search <query>
/ask <question>
/sources
/source <rank>
/source next
/source prev
/docs
/trace
/settings
/config
/clear
/exit
/quit
/q
```

Ordinary text and `/ask <question>` run Shell Ask in a background worker.
`/search <query>` runs Shell Search in a background worker. `/sources` and
`/source <rank|next|prev>` navigate the current source list shown in the
Inspector.

The TUI reads the default `.ragent` workspace from the current working
directory. It does not run ingest, build indexes, run retrieval eval, edit
config, or open local files.

Shell Ask does not write new traces in v0.1. CLI `ragent ask` remains the
trace-producing Ask workflow.

## Demo

See [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for a 3-5 minute demo script.

Short demo path:

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent chunks list --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent traces latest --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
uv run ragent tui
```

Demo screenshots:

![TUI Shell search](https://raw.githubusercontent.com/2017040393/ragent-forge/v0.1.0/docs/assets/v0_1/tui-shell-search.jpg)

![TUI source inspector](https://raw.githubusercontent.com/2017040393/ragent-forge/v0.1.0/docs/assets/v0_1/tui-source-inspector.jpg)

![TUI trace and settings](https://raw.githubusercontent.com/2017040393/ragent-forge/v0.1.0/docs/assets/v0_1/tui-trace-settings.jpg)

![Retrieval evaluation output](https://raw.githubusercontent.com/2017040393/ragent-forge/v0.1.0/docs/assets/v0_1/tui-retrieval-eval.jpg)

## Known Limitations

- Markdown/TXT are the only supported document formats.
- The lexical retriever is simple token overlap, not BM25.
- Semantic and hybrid retrieval require an embedding provider and a built vector
  index.
- The vector index is local JSONL, not a production vector database.
- Generation is optional and depends on an OpenAI Responses-compatible provider.
- With the default `null` provider, Ask stays in retrieval-only mode.
- Retrieval eval measures retrieval behavior only; it does not evaluate answer
  quality.
- Shell Ask does not write new traces.
- The TUI reads the default `.ragent` workspace from the current working
  directory.

## Non-Goals

v0.1.0 does not include:

- BM25.
- Reranking or cross-encoder reranking.
- Answer evaluation.
- LLM-as-judge.
- Query expansion.
- Multi-turn memory.
- Agent loops or planning loops.
- PDF or OCR support.
- Web UI.
- Vector databases such as Chroma, FAISS, or LanceDB.
- LangChain or LlamaIndex integration.
- OpenTelemetry.
- Streaming.
- Session persistence.
- TUI write operations such as ingest, index build, eval, or config editing.
- Source full-text viewer, local file opening, source table UI, or mouse source
  selection.

## Upgrade / Setup Notes

This is an early v0.1 release. No production migration path or runtime schema
migration is provided.

Recommended local setup:

```bash
uv sync --extra dev
uv run ragent ingest examples/knowledge --workspace .ragent
```

Semantic and hybrid retrieval require provider configuration and index build:

```bash
uv run ragent config init --workspace .ragent
uv run ragent index build --workspace .ragent
```

API keys are stored in local `.ragent/config.toml`; treat that file as
sensitive local state.

## Local Release Checklist

- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check .`.
- [ ] Run the demo commands from `docs/DEMO_SCRIPT.md`.
- [ ] Capture TUI screenshots if needed.
- [ ] Create tag `v0.1.0`.
- [ ] Push tag with `git push origin v0.1.0`.
- [ ] Copy these release notes into the GitHub Release.

This checklist is informational. This task does not create or push git tags.

## Suggested Next Versions

Future versions could improve retrieval quality, add richer source inspection,
introduce answer-quality evaluation, and explore a small explicitly controlled
agent layer. These are future directions, not v0.1.0 capabilities.
