# RAGentForge v0.1 Scope

> Language: English | [中文](V0_1_SCOPE.zh-CN.md)

## v0.1 Goal

RAGentForge v0.1 is a local-first, inspectable RAG MVP. Its goal is to make the
basic RAG workflow understandable and demo-ready: ingest local documents, chunk
them deterministically, retrieve relevant sources, optionally generate an
answer, inspect traces, run retrieval eval, and explore results through a
command-first TUI Shell.

## Included Capabilities

- Local Markdown/TXT document ingestion.
- Deterministic chunking.
- Local `.ragent/` workspace storage.
- JSONL chunk storage.
- Lexical retrieval.
- OpenAI-compatible semantic embeddings.
- Local JSONL vector index.
- Semantic retrieval.
- Hybrid RRF retrieval.
- Optional OpenAI Responses-compatible generation.
- Retrieval-only Ask mode with the default `null` provider.
- Answer sources.
- CLI operation traces.
- Retrieval evaluation with hit@k and MRR.
- Command-first Textual TUI Shell.
- Inline Shell command suggestions.
- Shell source navigation with `/sources`, `/source <rank>`, `/source next`,
  and `/source prev`.
- Selected-source Inspector.

## Explicit Non-Goals

v0.1 intentionally does not include:

- BM25.
- Reranking.
- Cross-encoder reranking.
- LLM-as-judge.
- Answer evaluation.
- Query expansion.
- Multi-turn memory.
- Agent tool loops.
- Planning loops.
- PDF support.
- OCR.
- Web UI.
- Vector databases.
- LangChain.
- LlamaIndex.
- Chroma, FAISS, or LanceDB.
- OpenTelemetry.
- Streaming.
- Session persistence.
- TUI ingest execution.
- TUI index build execution.
- TUI eval execution.
- TUI config editing.
- Source full-text viewer.
- Local file opening.
- Mouse source selection.

## What Makes This More Than a Toy Demo

- The workflow writes inspectable local artifacts instead of hiding state in
  memory.
- Chunks, vector indexes, traces, and eval reports are plain local files.
- Retrieval modes are explicit and testable.
- Hybrid retrieval records fusion metadata.
- Ask can run without generation, which makes retrieval behavior visible.
- CLI workflows and the TUI share application services instead of duplicating
  backend logic.
- The TUI is a real command shell with worker-backed Ask/Search, suggestions,
  source navigation, and an Inspector.
- Retrieval eval provides repeatable hit-rate and MRR checks over JSONL cases.

## Known Limitations

- Markdown/TXT are the only supported document formats.
- The lexical retriever is simple token overlap, not BM25.
- Semantic and hybrid retrieval require a configured embedding provider and a
  built vector index.
- The vector index is local JSONL, not a production vector database.
- Generation is optional and depends on an OpenAI Responses-compatible
  provider.
- Retrieval eval does not evaluate generated answer quality.
- Shell Ask does not write new traces; CLI `ragent ask` remains the
  trace-producing Ask workflow.
- The TUI is an inspectable shell, not a full management dashboard.

## v0.1 Readiness Checklist

- [x] Ingest local documents.
- [x] Inspect chunks.
- [x] Lexical search.
- [x] Semantic search.
- [x] Hybrid search.
- [x] Ask with sources.
- [x] Trace CLI Ask.
- [x] Retrieval eval.
- [x] Command-first TUI.
- [x] Source navigation.

## Suggested Future Roadmap

The roadmap below is future work, not current capability.

### v0.2 Retrieval Quality

- Improve lexical retrieval quality.
- Add retrieval comparison workflows.
- Improve source ranking inspection.
- Consider BM25 or reranking only after the current baseline is measured.

### v0.3 Answer Quality and Evaluation

- Add answer-quality evaluation.
- Add prompt comparison workflows.
- Track groundedness and citation quality.
- Keep answer eval separate from retrieval eval.

### v0.4 Agent Layer

- Add a small, explicit, user-controlled agent layer.
- Require visible plans, tool steps, and traces.
- Keep side effects opt-in and inspectable.
- Avoid hidden autonomous loops.
