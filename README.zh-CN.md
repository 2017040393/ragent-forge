# RAGentForge

> 语言: [English](README.md) | 中文

RAGentForge 是一个本地优先、可检查、command-first 的 RAG 控制台，用于在
终端里处理 Markdown/TXT 知识库。它关注 RAG 中最应该被看见的环节：导入
了什么、文本如何被切块、检索到了哪些来源、组装了什么 prompt，以及生成
了什么 trace。

## 它是什么

RAGentForge 是一个面向开发者的小型 Python 项目，适合在没有托管服务或隐
藏后端的情况下理解并演示端到端 retrieval augmented generation 工作流。
它把生成状态保存在本地 `.ragent/` workspace 中，并通过 CLI 命令和
Textual Shell TUI 暴露这些状态。

它不是完整的自主 Agent 框架。当前 v0.1 是一个本地、可检查的 MVP，覆盖
导入、检索、可选生成、trace、检索评估，以及 command-first TUI 检查。

## 为什么存在

很多 RAG demo 会把重要工程细节隐藏在托管应用、框架抽象或向量数据库后面。
RAGentForge 保持工作流朴素且可检查，让读者能看到数据如何从本地文档流向
chunks、检索结果、context pack、answer、sources、traces 和 eval reports。

## 功能

- 从本地文件或目录导入 Markdown/TXT。
- 生成确定性的 JSONL chunk 记录。
- 在 `.ragent/` 下保存本地 workspace 状态。
- 基于已生成 chunks 的 lexical retrieval。
- 面向 semantic retrieval 的 OpenAI-compatible embedding 配置。
- 用于 semantic search 的本地 JSONL vector index。
- 使用 Reciprocal Rank Fusion 融合 lexical 和 semantic 候选结果的 hybrid retrieval。
- 支持可选 OpenAI Responses-compatible generation 的 Ask pipeline。
- 未配置 generation 时的 retrieval-only Ask 模式。
- 带来源的回答和紧凑 source display。
- CLI ingest、index build、search、ask、retrieval eval 的本地 operation traces。
- 使用 hit@k 和 MRR 的 retrieval evaluation。
- 带 command suggestions、source navigation 和 Inspector panel 的 command-first Textual TUI Shell。

## 快速开始

安装开发依赖：

```bash
uv sync --extra dev
```

准备示例 workspace：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
uv run ragent chunks list --workspace .ragent
```

运行 lexical retrieval：

```bash
uv run ragent search "What is RAG?" --retrieval lexical --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

从项目根目录启动 command-first TUI：

```bash
uv run ragent tui
```

当前 TUI 会读取当前工作目录下默认的 `.ragent` workspace。导入、index build、
eval 和 config editing 仍通过 CLI 完成。

## 端到端 Demo

可复现 demo 流程见 [docs/PROJECT_WALKTHROUGH.zh-CN.md](docs/PROJECT_WALKTHROUGH.zh-CN.md)。

简版流程：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent chunks list --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent traces latest --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval lexical --workspace .ragent
uv run ragent tui
```

Semantic 和 hybrid retrieval 需要在 `.ragent/config.toml` 中配置 embedding
provider，并构建 vector index：

```bash
uv run ragent config init --workspace .ragent
uv run ragent index build --workspace .ragent
uv run ragent index status --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval semantic --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

默认配置下，generation 使用 `null` provider。在这种模式中，`ragent ask`
会检索并显示 context，但不会调用模型。

## 截图

带紧凑 source results 的 TUI Shell search：

![TUI Shell search](docs/assets/v0_1/tui-shell-search.jpg)

Source navigation 后的 selected-source Inspector：

![TUI source inspector](docs/assets/v0_1/tui-source-inspector.jpg)

TUI 中的 trace 和 settings inspection：

![TUI trace and settings](docs/assets/v0_1/tui-trace-settings.jpg)

Retrieval evaluation 输出：

![Retrieval evaluation output](docs/assets/v0_1/tui-retrieval-eval.jpg)

## Command-First TUI

`uv run ragent tui` 会打开一个 Shell 界面，包含 transcript、composer、
status line、inline command suggestions 和 selected-source Inspector。

普通文本和 `/ask <question>` 会在后台 worker 中运行 Shell Ask。
`/search <query>` 会在后台 worker 中运行 Shell Search。Shell 读取已有的
workspace chunks 和 indexes；它不会执行 ingest、构建 semantic index、运行
retrieval eval 或编辑 config。

常用 Shell commands：

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

Shell source navigation：

```text
/sources
/source <rank>
/source next
/source prev
```

输入 `/` 会打开 inline command candidates。使用 Up/Down 选择命令，再用
Tab 或 Enter 补全到 composer。命令执行仍通过 composer text 完成。

TUI 有意避免 `q` 这种全局单键快捷键。请在 composer 中输入 `/exit`、
`/quit` 或 `/q` 退出。

Shell Ask 在 v0.1 不写入新的 traces。CLI `uv run ragent ask ...` 仍然是
会产生 Ask trace 的 workflow。Shell `/trace` 读取 CLI workflow 已经写入的
最新 trace。

## 架构

项目围绕 presentation layers、application services、focused core modules
和 local workspace storage 组织。高层 pipeline 是：

```text
local documents
-> ingest
-> deterministic chunks
-> lexical / semantic / hybrid retrieval
-> context pack
-> optional generation
-> answer + sources
-> traces
-> retrieval eval
-> command-first TUI inspection
```

完整架构说明见 [docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md)。

TUI 设计说明见 [docs/TUI_COMMAND_SHELL_DESIGN.zh-CN.md](docs/TUI_COMMAND_SHELL_DESIGN.zh-CN.md)。

## 项目范围

当前范围见 [docs/V0_1_SCOPE.zh-CN.md](docs/V0_1_SCOPE.zh-CN.md)。

v0.1 包括本地导入、确定性 chunks、lexical retrieval、semantic retrieval、
hybrid RRF retrieval、optional generation、source display、traces、retrieval
evaluation 和 command-first TUI Shell。

## 发布与作品集材料

- [v0.1 Demo Script](docs/DEMO_SCRIPT.zh-CN.md)
- [v0.1 Release Notes](docs/RELEASE_NOTES_V0_1.zh-CN.md)
- [Portfolio Summary](docs/PORTFOLIO_SUMMARY.zh-CN.md)

## 当前限制

RAGentForge v0.1 有意不包含 BM25、reranking、cross-encoder reranking、
LLM-as-judge、answer evaluation、query expansion、multi-turn memory、agent
tool loops、planning loops、PDF/OCR、web UI、vector databases、streaming、
session persistence，也不包含 TUI ingest/index/eval/config editing 这类写操作。

Semantic 和 hybrid retrieval 需要 vector index。Generation 依赖配置好的
OpenAI Responses-compatible provider；否则 Ask 会保持 retrieval-only 模式。

## Roadmap

未来版本可能增加更好的 retrieval quality、answer evaluation、显式受控的
agent layers、更丰富的 source inspection，以及更完善的 developer ergonomics。
这些是未来方向，不是当前 v0.1 功能。

更多上下文：

- [docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md)
- [docs/PROJECT_WALKTHROUGH.zh-CN.md](docs/PROJECT_WALKTHROUGH.zh-CN.md)
- [docs/V0_1_SCOPE.zh-CN.md](docs/V0_1_SCOPE.zh-CN.md)
- [docs/TUI_COMMAND_SHELL_DESIGN.zh-CN.md](docs/TUI_COMMAND_SHELL_DESIGN.zh-CN.md)
- [docs/DEMO_SCRIPT.zh-CN.md](docs/DEMO_SCRIPT.zh-CN.md)
- [docs/RELEASE_NOTES_V0_1.zh-CN.md](docs/RELEASE_NOTES_V0_1.zh-CN.md)
- [docs/PORTFOLIO_SUMMARY.zh-CN.md](docs/PORTFOLIO_SUMMARY.zh-CN.md)
