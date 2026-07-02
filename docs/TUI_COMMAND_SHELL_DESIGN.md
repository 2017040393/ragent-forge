# TUI Command Shell Design

## Motivation

The TUI is a single command-first Shell interface rather than a management
dashboard. It should feel like a local RAG console: open it, type into the
composer, and keep the latest transcript output visible.

The direction is an Ask-first / command-first local RAG console. A user
should be able to open the TUI and immediately type a question. Secondary
workflows should be available through slash commands without navigation chrome
or global single-key shortcuts.

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
- `/q` also exits, as a slash command typed into the composer.

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

The shell layout should be compact and composer-first:

```text
Header/status
Transcript/output area
Bottom composer input
Optional Inspector/details panel
```

The Shell reuses the existing view models and formatters for read-only
summaries, retrieval, Ask, and selected-source details while keeping all primary
actions in the composer.

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

The shell state is modeled independently from Textual widgets:

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
from Textual rendering so it can support future Shell behavior without changing
backend services or CLI behavior.

## Command Dispatch

Command parsing should remain independent from UI rendering. The parser should
return a typed value such as `ParsedTuiCommand` with:

- command name
- args
- raw input
- slash-command flag
- optional error

The UI or a future dispatcher decides what to execute. This keeps parser tests
pure and makes it easier to extend Shell behavior without changing backend
services.

## Worker Behavior

Ask already needs worker execution to keep the current TUI responsive. The future
shell should also run Ask in a worker. Semantic and hybrid Search may also need
workers because query embedding can involve network latency.

Worker completion should append transcript messages and update shell state on
the UI thread. Worker failures should produce friendly error messages and never
display stack traces or API keys.

This design does not add new worker behavior by itself.

## Composer Polish

The composer should keep focus whenever input is enabled: on mount, after local
commands, after read-only command output, and after Ask or Search workers finish
or fail. While a worker is running, the input is disabled and should not be
force-focused.

Transcript updates should scroll to the latest output after local command
output, clear, worker start, worker completion, and worker failure. Source lists
should use compact, aligned labels with bounded width so long filenames do not
stretch the transcript. Inspector previews should stay compact and should show
only allowlisted retrieval metadata.

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
- TUI is now a single command-first Shell.
- Local Shell commands are wired.
- Read-only Shell commands `/docs`, `/trace`, and `/settings` are wired.
- Shell `/search <query>` is wired through a background worker.
- Shell search sources are displayed in the transcript.
- Shell Inspector shows selected-source details.
- Shell ordinary questions and `/ask <question>` are wired through a background
  worker.
- Lightweight inline command candidates are available while typing slash
  commands, with Up/Down selection and Tab/Enter completion into the composer.

## Migration Plan

The single Shell interface is the primary TUI surface. Future work should
improve the Shell itself: source navigation, command
suggestions, transcript polish, and optional richer status panels.

## Non-goals

- No command palette.
- No popup autocomplete.
- No command execution directly from the candidate list.
- No session persistence.
- No agent tool loop.
- No TUI ingest execution.
- No TUI index build execution.
- No TUI eval execution.
- No streaming.
- No config editing.
- No new backend features.
