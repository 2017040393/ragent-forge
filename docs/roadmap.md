# Roadmap

> Language: English | [中文](roadmap.zh-CN.md)

## v0.1: Local TUI + Inspectable RAG

Goals:
- Load local Markdown/TXT files.
- Chunk documents deterministically.
- Support lexical, semantic, and hybrid retrieval.
- Support source-grounded Ask with optional generation.
- Add local traces and retrieval evaluation.
- Show sources, traces, settings, and source inspection in the command-first TUI.

Non-goals:
- Real autonomous agents.
- Cloud sync or hosted services.
- PDF ingestion or complex document parsing.

## v0.2: Retrieval Quality + Better Source Inspection

Goals:
- Improve lexical retrieval quality beyond the current token-overlap baseline
  with BM25 as a stronger sparse baseline.
- Add retrieval comparison workflows.
- Make retrieval scores and source selection easier to inspect.
- Improve trace display, export, and demo polish.

Non-goals:
- Enterprise search features.
- Multi-user collaboration.
- Provider-specific lock-in.

## v0.3: Retrieval Quality and Efficiency Engineering

Primary outcomes:
- Improve retrieval efficiency, candidate recall, and final-result precision
  against frozen, reproducible baselines.
- Make query processing, candidate retrieval, deduplication, ranking, and
  context selection explicit and inspectable without choosing an implementation
  before the measurements are agreed.
- Provide one retrieval entry point over typed items. Document evidence,
  workspace-local project facts, user notes, and session memory keep distinct
  provenance and lifecycle semantics even when they share a retrieval pipeline.
- Keep workspace-local project memory editable and auditable; it may become the
  primary user-facing knowledge surface without erasing document evidence.
- Extend traces and evaluation so quality, latency, context cost, and source
  behavior can be diagnosed at each retrieval stage.

Measurement protocol:
- Freeze a benchmark manifest before comparing implementations. It records the
  corpus and eval-set versions, workspace configuration, runtime and hardware
  profile, retrieval limits, context budget, and cold-versus-warm run policy.
- Reuse the versioned v0.2 retrieval eval as the initial benchmark instead of
  creating parallel document-only, memory-only, and mixed-source suites before
  those product behaviors exist. Keep exact-term and paraphrase coverage, and
  add other query categories only when they represent a declared v0.3 behavior.
- Use the v0.2 hybrid retriever and its current document evidence corpus as the
  initial baseline for the shared retrieval entry point.
- Add focused memory correctness cases when project memory is introduced. Add
  mixed-source cases only when the product claims cross-source retrieval,
  conflict resolution, or combined context behavior.
- Run each measured configuration at least three times. Report cold and warm
  results separately and persist machine-readable artifacts for every run.
- Measure candidate `hit@k` and `recall@k`; final `precision@k`, `MRR`, and
  `nDCG@k`; retrieval latency `p50` and `p95`; candidate counts; selected
  context characters or tokens; and failures by source type and pipeline stage.

Initial release gates:
- No retrieval improvement technique is selected for v0.3 until the benchmark
  manifest and v0.2 baseline report are checked in.
- A quality-oriented configuration improves document `recall@20` and final
  `precision@5` by at least 5 percentage points over the v0.2 hybrid baseline.
  Its `MRR` and `nDCG@10` may not regress by more than 1 percentage point, warm
  `p95` retrieval latency must stay within 1.5x of baseline, and average selected
  context tokens may not increase.
- An efficiency-oriented configuration keeps `recall@20`, `precision@5`, and
  `nDCG@10` within 1 percentage point of its frozen baseline while reducing warm
  `p95` retrieval latency by at least 20% and average selected context tokens by
  at least 15%.
- No declared query-category or source-kind slice with enough cases for a stable
  comparison may regress by more than 3 percentage points on `recall@20` or
  `precision@5` without a documented release exception.
- The recommended default must not be strictly worse than its baseline on both
  retrieval quality and latency, and all release-gate results must be
  reproducible from checked-in configuration and eval artifacts.
- Targets may be revised after the first baseline run, but they must be frozen
  before implementation results are used to choose a technical approach.

Non-goals:
- Hidden long-term memory.
- Cloud profiles.
- Automatic ingestion of unrelated files.
- Agent-directed iterative or autonomous multi-step retrieval.

## v0.4: Minimal Agent Runtime

Goals:
- Add a small controlled runtime for explicit multi-step workflows.
- Build planned query refinement and iterative retrieval on the inspectable
  retrieval pipeline established in v0.3.
- Require visible plans, tool steps, and trace output.
- Keep the user in control of side effects.

Non-goals:
- Fully autonomous background agents.
- Browser automation.
- Distributed task execution.

## v0.5: Evaluation Dashboard

Goals:
- Track retrieval quality and answer quality over local test sets.
- Add simple comparison views for prompts, retrieval stages and settings, and
  document-versus-memory source behavior.
- Support learning-oriented experiments.

Non-goals:
- Enterprise observability.
- Hosted analytics.
- Complex model evaluation infrastructure.

## v0.6: Open-source Polish

Goals:
- Improve documentation and examples.
- Stabilize public interfaces.
- Add contributor guidance and release workflows.

Non-goals:
- Large marketplace or plugin ecosystem.
- Desktop packaging.
- Production SaaS features.
