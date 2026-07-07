# TUI Command Shell Design

> 语言: [English](TUI_COMMAND_SHELL_DESIGN.md) | 中文

## 动机

TUI 是一个单一的 command-first Shell 界面，而不是管理 dashboard。它应该像
一个本地 RAG console：打开它，在 composer 中输入，并保持最新 transcript
输出可见。

方向是 Ask-first / command-first 的本地 RAG console。用户应该能打开 TUI 后
立刻输入问题。次要 workflows 应通过 slash commands 可用，而不需要导航
chrome 或全局单键快捷键。

## 目标交互模型

普通输入表示 Ask。如果 composer 中包含普通文本，shell 应使用当前 retrieval
settings 运行现有 Ask flow。

Slash commands 控制显式 workflows：

- `/ask <question>` 显式运行 Ask。
- `/search <query>` 运行 retrieval search。
- `/sources` 显示当前 source list。
- `/source <rank|next|prev>` 切换 Inspector 中显示的 source。
- `/docs` 显示 workspace 和 document summary。
- `/trace` 显示最新 trace。
- `/settings` 显示只读 config summary。
- `/mode lexical|bm25|semantic|hybrid` 修改当前 retrieval mode。
- `/limit <n>` 修改 retrieval result limit。
- `/context <n>` 修改 Ask 的 max context chars。
- `/prompt on|off` 切换 prompt preview。
- `/help` 显示可用 commands。
- `/clear` 清空 transcript。
- `/exit` 和 `/quit` 退出 TUI。
- `/q` 也会退出，它是 typed into the composer 的 slash command。

示例：

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

Shell layout 应紧凑并以 composer 为中心：

```text
Header/status
Transcript/output area
Bottom composer input
Optional Inspector/details panel
```

Shell 复用现有 view models 和 formatters 来呈现 read-only summaries、
retrieval、Ask 和 selected-source details，同时把所有主要 actions 保持在
composer 中。

## Slash Commands

MVP command registry 应保持保守：

| Command | Aliases | Behavior |
| --- | --- | --- |
| `/help` | | Show available commands. |
| `/ask <question>` | | Run Ask explicitly. |
| `/search <query>` | `/s` | Search chunks. |
| `/sources` | | Show current sources. |
| `/source <rank|next|prev>` | | Select a source by rank, next, or prev. |
| `/docs` | | Show document summary. |
| `/trace` | `/t` | Show the latest trace. |
| `/settings` | `/config` | Show read-only config summary. |
| `/mode lexical|bm25|semantic|hybrid` | | Set retrieval mode. |
| `/limit <n>` | | Set retrieval result limit. |
| `/context <n>` | | Set max context chars. |
| `/prompt on|off` | | Toggle prompt preview. |
| `/clear` | | Clear transcript. |
| `/exit` | `/quit`, `/q` | Exit the TUI. |

Aliases 应保持稀疏且可预测。第一个 shell MVP 不应增加 fuzzy matching。

## Shell State

Shell state 独立于 Textual widgets 建模：

```python
retrieval_mode: "lexical" | "bm25" | "semantic" | "hybrid"
limit: int
max_context_chars: int
show_prompt: bool
running: bool
messages: list[TranscriptMessage]
selected_source: SearchResult | None
available_sources: list[TranscriptSource]
```

该 state 应由 command dispatch 和 worker completion 更新，而不是由 parser code
更新。

## Transcript Model

Transcript 在一个 session 中应是 append-only。最小 message model 是：

```python
role: "system" | "user" | "assistant" | "tool" | "error"
text: str
metadata: dict[str, Any]
```

用户问题、command acknowledgements、generated answers、search summaries 和
friendly errors 都可以成为 transcript messages。Metadata 可以携带 selected
source ids、retrieval mode、limits 或 trace ids，而不强迫这些细节进入可见文本。

Transcript model foundation 位于 `src/ragent_forge/tui/shell_models.py`。它有
意保持纯粹并独立于 Textual rendering，以便支持未来 Shell 行为，而不改变
backend services 或 CLI behavior。

## Command Dispatch

Command parsing 应独立于 UI rendering。Parser 应返回类型化值，例如
`ParsedTuiCommand`，包含：

- command name
- args
- raw input
- slash-command flag
- optional error

UI 或未来 dispatcher 决定执行什么。这让 parser tests 保持纯净，也让扩展
Shell 行为时不必改 backend services。

## Worker Behavior

Ask 和 Search 在 workers 中运行，以保持当前 TUI 响应。Semantic 和 hybrid
Search 可能涉及 query embedding 网络延迟，因此 worker execution 可以保持
composer 和 transcript 响应。

Worker completion 应在 UI thread 上 append transcript messages 并更新 shell
state。Worker failures 应产生友好的 error messages，绝不显示 stack traces 或
API keys。Shell Search 和 Shell Ask 是基于当前本地 workspace 的 read-oriented
TUI workflows；CLI commands 仍然是 trace-producing workflows。

## Composer Polish

只要 input 可用，composer 就应保持 focus：mount 时、本地 commands 后、
read-only command output 后、Ask 或 Search workers 完成或失败后。Worker 运行
时 input 会被禁用，不应强行 focus。

Transcript updates 应在 local command output、clear、worker start、worker
completion 和 worker failure 后滚动到最新输出。Source lists 应使用紧凑、对齐
且有宽度边界的 labels，避免长文件名撑开 transcript。Inspector previews 应保
持紧凑，并只显示 allowlisted retrieval metadata。

## 复用现有 Services

Shell 应复用现有 TUI view-model functions 和 application services：

- `run_tui_ask`
- `run_tui_search`
- `load_documents_page_model`
- `load_trace_page_model`
- `load_settings_page_model`
- existing formatters where useful

Shell 不应重复 ingestion、indexing、retrieval、generation、trace 或 config logic。

## 实现状态

当前实现状态：

- Command parser 已存在。
- Transcript model 已存在。
- TUI 现在是单一 command-first Shell。
- Local Shell commands 已接入。
- Read-only Shell commands `/docs`、`/trace` 和 `/settings` 已接入。
- Shell `/search <query>` 已通过 background worker 接入。
- Shell search sources 会显示在 transcript 中。
- Shell Inspector 会显示 selected-source details。
- Shell source navigation commands `/sources` 和 `/source <rank|next|prev>` 已接入。
- Shell ordinary questions 和 `/ask <question>` 已通过 background worker 接入。
- Typing slash commands 时有 lightweight inline command candidates，可用 Up/Down
  选择，并用 Tab/Enter 补全到 composer。

## Migration Plan

单一 Shell interface 是主要 TUI surface。未来工作应改进 Shell 本身：更丰富的
source inspection、transcript polish 和可选的 richer status panels。

## Non-goals

- 无 command palette。
- 无 modal popup autocomplete；当前 Shell 在 composer area 使用 inline command candidates。
- 不从 candidate list 直接执行 command。
- 不打开本地文件。
- 无 source table UI。
- 无 session persistence。
- 无 agent tool loop。
- 无 TUI ingest execution。
- 无 TUI index build execution。
- 无 TUI eval execution。
- 无 streaming。
- 无 config editing。
- 无新的 backend features。
