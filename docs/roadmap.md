# Roadmap

## v0.1: Local TUI + Inspectable RAG

Goals:
- Load local Markdown/TXT files.
- Chunk documents deterministically.
- Add a minimal local index and retrieval path.
- Show sources and traces in the TUI.

Non-goals:
- Real autonomous agents.
- Cloud sync or hosted services.
- PDF ingestion or complex document parsing.

## v0.2: Hybrid Retrieval + Better Trace

Goals:
- Combine keyword and vector retrieval.
- Make retrieval scores and source selection easier to inspect.
- Improve trace display and export.

Non-goals:
- Enterprise search features.
- Multi-user collaboration.
- Provider-specific lock-in.

## v0.3: Project Memory

Goals:
- Add workspace-local memory for project facts and user-curated notes.
- Keep memory editable and auditable.
- Distinguish retrieved evidence from remembered project context.

Non-goals:
- Hidden long-term memory.
- Cloud profiles.
- Automatic ingestion of unrelated files.

## v0.4: Minimal Agent Runtime

Goals:
- Add a small controlled runtime for explicit multi-step workflows.
- Require visible plans, tool steps, and trace output.
- Keep the user in control of side effects.

Non-goals:
- Fully autonomous background agents.
- Browser automation.
- Distributed task execution.

## v0.5: Evaluation Dashboard

Goals:
- Track retrieval quality and answer quality over local test sets.
- Add simple comparison views for prompts and retrieval settings.
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
