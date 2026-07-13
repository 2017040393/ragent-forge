# v0.3 Architecture Convergence

> Language: English | [中文](V0_3_ARCHITECTURE_CONVERGENCE.zh-CN.md)

## Purpose

This document defines the implementation sequence and acceptance criteria for
closing the architecture risks identified before v0.3 retrieval experiments.
The work is one coordinated refactor, delivered as independently green commits.
It does not select a concrete reranking algorithm or ANN database before the
v0.3 benchmark justifies one.

## Target Dependency Direction

```text
CLI / TUI
  -> application use cases
  -> RetrievalEngine and application ports
  -> core contracts

composition root
  -> application ports
  -> infrastructure adapters
     - local generation workspace
     - prepared sparse and dense indexes
     - OpenAI-compatible providers
     - trace repository
```

Application and core modules must not import infrastructure modules. The
composition root is the only place that selects concrete adapters.

## Retrieval Runtime

Every retrieval entry point uses one `RetrievalEngine.run()` contract:

```text
strict mode parsing
-> query processing
-> candidate retrieval
-> deduplication
-> optional reranking
-> context selection
-> RetrievalRun persistence
```

Each stage is an injected protocol with typed input and output. Retrieval modes
select candidate adapters; they are not separate end-to-end pipelines. A no-op
reranker is valid, but it must be explicit and replaceable. Search, Ask,
retrieval eval, comparison, CLI, and TUI must all consume the same
`RetrievalRun` result.

## Typed Data Boundary

Core retrieval data uses typed models for:

- source identity and source kind;
- provenance, authority, freshness, and lifecycle;
- retrieval items and candidates;
- stage inputs, outputs, timings, and failures;
- selected context and trace references.

Unvalidated dictionaries are allowed only at JSON, TOML, HTTP, and legacy-file
boundaries. Boundary parsers must convert them to typed contracts before
application logic consumes them.

## Workspace Transaction Model

Workspace generation state is immutable after commit:

```text
.ragent/
  current.json
  generations/<snapshot-id>/
    manifest.json
    chunks.jsonl
    ingest_summary.json
    vector_index.jsonl
    vector_index_manifest.json
  traces/
  eval/
  sessions/
```

Writers create a complete generation in a temporary directory, validate it,
atomically rename it into `generations/`, and atomically replace `current.json`
last. A failed write leaves the previous generation readable. Traces, eval
runs, and sessions are append-oriented and reference a committed snapshot id.

Legacy flat workspaces remain readable. A migration registry upgrades known
schemas explicitly and supports dry-run inspection. Unknown or future schemas
fail with an actionable error; no retrieval mode or schema silently falls back.

## Prepared Retrieval State

Prepared state is keyed by snapshot id:

- chunks are parsed once per process and snapshot;
- lexical tokenization is prepared once;
- BM25 document frequencies and lengths are prepared once;
- vector records and chunk lookup maps are loaded once;
- a snapshot change invalidates all prepared state.

Exact vector scan remains a supported adapter for the current corpus. The port
must permit a later ANN adapter. Cold and warm timings are measured separately;
the refactor does not claim unbounded sublinear performance without an ANN
benchmark.

## Unified Observability

`RetrievalRun` is the canonical retrieval trace payload. CLI and TUI persist the
same operation trace shape. TUI sessions store a `trace_id` and compact display
metadata instead of inventing a second retrieval trace format. Evaluation keeps
per-case stage timings and aggregates them into stage-level percentiles.

## Responsibility Split

The refactor splits ownership by use case rather than by arbitrary line count:

- `cli/__init__.py`: parser facade and top-level dispatch only;
- `cli/parser.py`: argument parser construction;
- `cli/handlers/`: ingest, index, retrieval, eval, config, and trace handlers;
- `tui/main.py`: Textual composition and event routing only;
- `tui/controllers/`: Ask, search, session, and worker coordination;
- retrieval eval: `evaluation/contracts.py`, `cases.py`, `runner.py`,
  `metrics.py`, and `reporting.py`;
- infrastructure: filesystem, provider, and prepared-index adapters.

The old import paths may remain as thin compatibility facades, but production
application code must use the new ownership boundaries.

## Completion Evidence

The convergence sequence is implemented through independently green commits.
The final presentation/evaluation split is covered by architecture and
compatibility tests. The checked-in benchmark manifest is
`benchmarks/prepared_retrieval_manifest.json`; run it with:

```text
uv run --extra dev python -m benchmarks.prepared_retrieval
```

The benchmark reports cold and warm timings separately and gates structural
cache reuse. It deliberately does not claim a retrieval-quality improvement or
ANN scalability result. Retrieval evaluation reports now include typed
stage-level latency summaries (`sample_count`, `average_ms`, `p50_ms`, and
`p95_ms`) in addition to overall latency metrics.

## Acceptance Matrix

| Risk | Completion criterion |
| --- | --- |
| Retrieval remains a black box | Every entry point returns `RetrievalRun`; stages are injected and independently tested. |
| Metadata is loosely typed | Domain and application models avoid `dict[str, Any]`; untyped mappings remain only in boundary parsers. |
| Atomic files are not a transaction | Fault injection at every generation write point leaves the previous snapshot readable. |
| Infrastructure boundary is conceptual | Architecture tests reject application/core imports of infrastructure; providers and filesystem code live under infrastructure. |
| Query cost repeats work | Warm queries reuse snapshot-keyed prepared chunks, BM25 state, and vector records. |
| CLI/TUI observability differs | Both surfaces persist the same retrieval trace schema and TUI sessions reference its `trace_id`. |
| Large files concentrate responsibilities | CLI, TUI, and eval ownership is split into focused modules with compatibility tests. |
| Invalid modes become lexical | Invalid values raise a clear error at every boundary. |
| Schema has no migration | Versioned migration registry and golden legacy fixtures cover all supported artifact upgrades. |

## Delivery Sequence

1. Freeze the acceptance matrix and architecture rules.
2. Introduce typed contracts and strict parsing.
3. Route every retrieval use case through `RetrievalEngine`.
4. Introduce generation directories and schema migrations.
5. finish infrastructure and composition boundaries.
6. Add snapshot-keyed prepared retrieval state and performance assertions.
7. Unify operation traces and TUI trace references.
8. Split presentation and evaluation modules, update docs, and run the full
   verification and benchmark suite.

Every step must pass Pyright, Ruff, and relevant tests before it is committed
and pushed. The full suite runs again at the final step.
