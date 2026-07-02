# TUI Command Shell Design

## Motivation

The current page-based TUI is functional: it exposes Documents, Search, Ask,
Trace, Settings, and an Inspector. As the feature set grows, that structure
increasingly feels like a management dashboard rather than a local RAG console.

The future direction is an Ask-first / command-first local RAG console. A user
should be able to open the TUI and immediately type a question. Secondary
workflows should be available through slash commands instead of forcing the user
to move through page navigation first.

## Target Interaction Model

Normal input means Ask. If the composer contains ordinary text, the shell should
run the existing Ask flow with the current retrieval settings.

Slash commands control explicit workflows:

- `/ask <question>` explicitly runs Ask.
- `/search <query>` runs retrieval search.
- `/docs` shows workspace and document summary.
- `/trace` shows the latest trace.
- `/settings` shows a read-only config summary.
- `/mode lexical|semantic|hybrid` changes the current retrieval mode.
- `/limit <n>` changes the retrieval result limit.
- `/context <n>` changes max context chars for Ask.
- `/prompt on|off` toggles prompt preview.
- `/help` shows available commands.
- `/clear` clears the transcript.
- `/exit` and `/quit` exit the TUI.

Example:

```text
> /mode hybrid
retrieval mode set to hybrid

> What is Agentic RAG?
Running ask...

Answer:
  ...

Sources:
  1. rag_basics.md score=...
```

## Proposed Layout

The future shell layout should be compact and composer-first:

```text
Header/status
Transcript/output area
Bottom composer input
Optional Inspector/details panel
```

The existing three-column page UI should not be deleted immediately. The command
shell should be added incrementally, proven with tests, and allowed to reuse the
current page view models and formatters while the project learns which shell
interactions are worth promoting.

## Slash Commands

The MVP command registry should stay conservative:

| Command | Aliases | Behavior |
| --- | --- | --- |
| `/help` | | Show available commands. |
| `/ask <question>` | | Run Ask explicitly. |
| `/search <query>` | `/s` | Search chunks. |
| `/docs` | | Show document summary. |
| `/trace` | `/t` | Show the latest trace. |
| `/settings` | `/config` | Show read-only config summary. |
| `/mode lexical|semantic|hybrid` | | Set retrieval mode. |
| `/limit <n>` | | Set retrieval result limit. |
| `/context <n>` | | Set max context chars. |
| `/prompt on|off` | | Toggle prompt preview. |
| `/clear` | | Clear transcript. |
| `/exit` | `/quit`, `/q` | Exit the TUI. |

Aliases should remain sparse and predictable. The parser should not add fuzzy
matching in the first shell MVP.

## Shell State

The future shell state can be modeled independently from Textual widgets:

```python
retrieval_mode: "lexical" | "semantic" | "hybrid"
limit: int
max_context_chars: int
show_prompt: bool
running: bool
messages: list[TranscriptMessage]
selected_source: SearchResult | None
```

This state should be updated by command dispatch and worker completion, not by
parser code.

## Transcript Model

The transcript should be append-only during a session. A minimal message model is:

```python
role: "system" | "user" | "assistant" | "tool" | "error"
text: str
metadata: dict[str, Any]
```

User questions, command acknowledgements, generated answers, search summaries,
and friendly errors can all become transcript messages. Metadata can carry
selected source ids, retrieval mode, limits, or trace ids without forcing those
details into visible text.

The transcript model foundation lives in
`src/ragent_forge/tui/shell_models.py`. It is intentionally pure and independent
from Textual rendering so it can support a future Shell page without changing
the existing page-based TUI.

## Command Dispatch

Command parsing should remain independent from UI rendering. The parser should
return a typed value such as `ParsedTuiCommand` with:

- command name
- args
- raw input
- slash-command flag
- optional error

The UI or a future dispatcher decides what to execute. This keeps parser tests
pure and makes it easier to add a Shell page without changing backend services.

## Worker Behavior

Ask already needs worker execution to keep the current TUI responsive. The future
shell should also run Ask in a worker. Semantic and hybrid Search may also need
workers because query embedding can involve network latency.

Worker completion should append transcript messages and update shell state on
the UI thread. Worker failures should produce friendly error messages and never
display stack traces or API keys.

This design does not add new worker behavior by itself.

## Reusing Existing Services

The shell should reuse existing TUI view-model functions and application
services:

- `run_tui_ask`
- `run_tui_search`
- `load_documents_page_model`
- `load_trace_page_model`
- `load_settings_page_model`
- existing formatters where useful

The shell should not duplicate ingestion, indexing, retrieval, generation, trace,
or config logic.

## Implementation Status

Current implementation status:

- Command parser exists.
- Transcript model exists.
- Experimental Shell page exists.
- Local Shell commands are wired.
- Read-only Shell commands `/docs`, `/trace`, and `/settings` are wired.
- Shell `/search <query>` is wired through a background worker.
- Shell search sources are displayed in the transcript.
- Shell Inspector shows selected-source details.
- Shell ordinary questions and `/ask <question>` are wired through a background
  worker.

## Migration Plan

1. Add command parser and design doc.
2. Add transcript models.
3. Add a Shell page while keeping existing pages.
4. Make Shell the default TUI page after the interaction is proven.
5. Gradually reduce reliance on left navigation if desired.
6. Add optional command suggestions later.

Each phase should keep existing pages working until there is a tested replacement
for the relevant workflow.

## Non-goals

- No autocomplete popup in the first shell MVP.
- No session persistence.
- No agent tool loop.
- No TUI ingest execution.
- No TUI index build execution.
- No TUI eval execution.
- No streaming.
- No config editing.
- No new backend features.
