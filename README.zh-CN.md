# RAGentForge

> 语言: [English](README.md) | 中文

RAGentForge 是一个本地优先、command-first 的 RAG workbench，用于构建、
检查和评估 retrieval-augmented systems。它关注 retrieval quality、grounded
context construction、traceable runs、reproducible evaluation，以及
Markdown/TXT/PDF 知识库上的 retrieval mode comparison。

## 它是什么

RAGentForge 是一个面向开发者的小型 Python 项目，适合在没有托管服务或隐
藏后端的情况下理解、调试并演示 retrieval augmented generation 工作流。
它把生成状态保存在本地 `.ragent/` workspace 中，并通过 CLI 命令和
Textual Shell TUI 暴露这些状态。

它不是完整的自主 Agent 框架。当前 v0.2 是一个本地、可检查的基础版本，
覆盖导入、检索、可选生成、trace、span-grounded retrieval evaluation、
retrieval comparison，以及 command-first TUI 检查。

## 为什么存在

很多 RAG demo 会把重要工程细节隐藏在托管应用、框架抽象或向量数据库后面。
RAGentForge 保持工作流朴素且可检查，让读者能看到数据如何从本地文档流向
chunks、检索结果、context pack、answer、sources、traces 和 eval reports。

## 功能

- 从本地文件或目录导入 Markdown/TXT/PDF。
- 通过 `DocumentBlock[] -> BlockChunker` 统一结构化导入 Markdown、TXT 和 PDF。
- 生成带格式感知 metadata 的确定性 JSONL chunk 记录。
- 在 `.ragent/` 下保存本地 workspace 状态。
- 基于已生成 chunks 的 lexical 和 BM25 retrieval。
- 面向 semantic retrieval 的 OpenAI-compatible embedding 配置。
- 用于 semantic search 的本地 JSONL vector index。
- 使用 Reciprocal Rank Fusion 融合 BM25 和 semantic 候选结果的 hybrid retrieval。
- 支持可选 OpenAI Responses-compatible generation 的 Ask pipeline。
- 未配置 generation 时的 retrieval-only Ask 模式。
- 带来源的回答和紧凑 source display。
- CLI ingest、index build、search、ask、retrieval eval 的本地 operation traces。
- 从稳定 source evidence 生成 span-based synthetic eval cases。
- 使用 Hit@k、Recall@k、MRR、latency 和 context-size metrics 的 retrieval evaluation。
- 持久化 retrieval eval run reports，包含紧凑 cases 和 failures JSONL。
- 使用 `failure_type` 和 `failure_reason` 的确定性 failure analysis。
- 对 lexical、BM25、semantic 和 hybrid retrieval modes 进行 compare。
- 带默认 hybrid Ask、流式回答显示、saved sessions、source navigation 和 Inspector
  panel 的 command-first Textual TUI Shell。

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

运行 lexical 或 BM25 retrieval：

```bash
uv run ragent search "What is RAG?" --retrieval lexical --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval bm25 --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
```

从项目根目录启动 command-first TUI：

```bash
uv run ragent tui
uv run ragent tui --workspace .ragent
```

不加 `--workspace` 时，TUI 会读取当前工作目录下默认的 `.ragent` workspace。
如果要检查其他本地 workspace，可以传 `--workspace`。TUI sessions 和 exports
会保存在该 workspace 的 `sessions/` 目录下；默认 workspace 时就是
`.ragent/sessions/`。导入、index build、eval 和 config editing 仍通过 CLI 完成。

## 端到端 Demo

可复现 demo 流程见 [docs/PROJECT_WALKTHROUGH.zh-CN.md](docs/PROJECT_WALKTHROUGH.zh-CN.md)。

简版流程：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent chunks list --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval bm25 --workspace .ragent
uv run ragent ask "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent traces latest --workspace .ragent
uv run ragent eval retrieval --cases examples/eval/retrieval_cases.jsonl --retrieval bm25 --workspace .ragent --limit 5
uv run ragent eval compare --cases examples/eval/retrieval_cases.jsonl --retrieval lexical,bm25 --limit 1,3,5 --workspace .ragent
uv run ragent tui --workspace .ragent
```

## v0.2 Retrieval Quality Foundation

目标：让 retrieval quality 可以被测量、比较和诊断。

核心思想是 eval dataset 不应该和某一次具体 chunking strategy 绑死。
RAGentForge 可以生成和评估 span-grounded retrieval eval cases。Evidence spans
会在 evaluation 时映射到当前 chunk index，因此即使 chunking strategy 变化，
eval cases 仍然稳定。

v0.2 包括：

- Span-based synthetic eval generation。
- Evidence-to-current-chunk mapping。
- 带 Hit@k、Recall@k、MRR、latency 和 context-cost metrics 的 retrieval eval runner。
- `.ragent/eval/runs/` 下的持久化 eval run reports。
- 确定性 failure analysis。
- Lexical、BM25、semantic 和 hybrid retrieval comparison。
- 便于 review 和自动化的本地 JSON/JSONL artifacts。

完整 workflow、metrics、artifacts、failure types 和 demo script 见
[docs/RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md)。

## Retrieval Evaluation Workflow

RAGentForge 可以从稳定 source evidence spans 生成 retrieval evaluation cases，
而不是只依赖手写固定 chunk ids。这样 dataset 可以跨 chunk-size、
chunk-overlap、retrieval-mode 和 ranking 变化复用：cases 指向 source evidence，
`eval retrieval` 再把 evidence 映射回当前 workspace chunks。

紧凑本地流程：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --dry-run
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --overwrite
uv run ragent eval retrieval --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval bm25 --limit 5
uv run ragent eval compare --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5
```

`eval generate --dry-run` 不会调用模型。非 dry-run 生成需要配置 generation
provider。需要从 text-based PDFs 生成 cases 时添加 `--include-pdf`。

上面的 compare 命令包含 semantic 和 hybrid runs。如果希望这些 runs 成功，
需要先构建 vector index；lexical 和 BM25 不需要 embeddings。

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

## Retrieval Modes

- `lexical`：简单 token-overlap baseline。
- `bm25`：使用 BM25 scoring 的更强 lexical baseline，不需要 vector index。
- `semantic`：embedding-based vector retrieval，需要先运行
  `uv run ragent index build --workspace .ragent`。
- `hybrid`：BM25 和 semantic retrieval 的 Reciprocal-rank-fusion style
  组合，同样需要 vector index。

示例 compare 输出如下；数字只是说明格式，不是 checked-in benchmark results：

```text
mode      k   status   hit@k   recall@k   mrr     avg_latency_ms   failures
lexical   5   success  0.5000  0.4200     0.3900  3.2000           4
bm25      5   success  0.6500  0.5700     0.5100  4.1000           3
semantic  5   success  0.7000  0.6200     0.5600  18.3000          2
hybrid    5   success  0.7800  0.6900     0.6300  22.5000          1
```

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

`uv run ragent tui --workspace .ragent` 会打开一个 Shell 界面，包含聚焦于
user/assistant 的 transcript、composer、status line、inline command suggestions、
source/session pickers，以及用于 selected answer 或 selected source 的 Inspector。

普通文本默认就是 Ask，所以直接输入问题等价于 `/ask <question>`。默认 retrieval
mode 是 `hybrid`；如果配置的 provider 支持 streaming，生成回答会流式写入
transcript。`/search <query>` 会在后台 worker 中运行 Shell Search。Shell 读取
已有的 workspace chunks 和 indexes；它不会执行 ingest、构建 semantic index、
运行 retrieval eval 或编辑 config。

TUI 启动时会恢复最近保存的 session。成功或失败的 Ask 都会作为 assistant turn
保存，并绑定 sources 和 run metadata，位置在 `.ragent/sessions/`。Inspector 会
跟随 selected answer，因此可以用 `/turn`、`/source` 和 source picker 查看某条
回答对应的 evidence。

常用 Shell commands：

```text
/help
/mode lexical|bm25|semantic|hybrid
/limit <n>
/context <n>
/prompt on|off
/search <query>
/ask <question>
/sources
/source <rank>
/source next
/source prev
/sessions
/new
/switch <session-id>
/rename <title>
/delete
/pin
/star
/session-search <query>
/export markdown|json
/branch
/rerun
/continue-sources
/title [auto|text]
/turn <id|number|next|prev|first|last>
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

Shell Ask 会写入 TUI session artifacts，但不会写入 operation traces。CLI
`uv run ragent ask ...` 仍然是会产生 Ask trace 的 workflow。Shell `/trace`
读取 CLI workflow 已经写入的最新 trace。

## 架构

项目围绕 presentation layers、application services、focused core modules
和 local workspace storage 组织。高层 pipeline 是：

```text
local documents
-> ingest
-> deterministic chunks
-> lexical / BM25 / semantic / hybrid retrieval
-> context pack
-> optional generation
-> answer + sources
-> traces
-> retrieval eval
-> command-first TUI inspection
-> saved TUI sessions and exports
```

完整架构说明见 [docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md)。

TUI 设计说明见 [docs/TUI_COMMAND_SHELL_DESIGN.zh-CN.md](docs/TUI_COMMAND_SHELL_DESIGN.zh-CN.md)。

## 项目范围

当前范围见 [docs/V0_1_SCOPE.zh-CN.md](docs/V0_1_SCOPE.zh-CN.md)。

v0.1 包括本地导入、确定性 chunks、lexical retrieval、semantic retrieval、
hybrid RRF retrieval、optional generation、source display、traces、retrieval
evaluation 和 command-first TUI Shell。

v0.2 增加 retrieval quality foundation：span-grounded eval generation、
evidence-to-chunk mapping、更丰富的 retrieval metrics、持久化 eval run
reports、failure analysis、retrieval comparison 和 BM25。

当前 TUI 还包含本地 session workbench：saved conversations、pin/star/search、
session export、branch/rerun helpers、selected-answer source inspection 和
streaming Ask output。

## 发布与作品集材料

- [v0.1 Demo Script](docs/DEMO_SCRIPT.zh-CN.md)
- [v0.2 Retrieval Evaluation Guide](docs/RETRIEVAL_EVALUATION.zh-CN.md)
- [v0.2 Release Notes](docs/RELEASE_NOTES_V0_2.zh-CN.md)
- [v0.1 Release Notes](docs/RELEASE_NOTES_V0_1.zh-CN.md)
- [Portfolio Summary](docs/PORTFOLIO_SUMMARY.zh-CN.md)

## 当前限制

RAGentForge v0.2 有意不包含 reranking、cross-encoder reranking、query
rewriting、agentic multi-step retrieval、LLM-as-judge answer grading、RAGAS
integration、OCR/scanned PDF support、PDF viewing/editing、web dashboard、
vector databases、agent tool loops，也不包含 TUI ingest/index/eval/config
editing 这类写操作。

Semantic 和 hybrid retrieval 需要 vector index。Generation 依赖配置好的
OpenAI Responses-compatible provider；否则 Ask 会保持 retrieval-only 模式。

## Roadmap

未来版本可能增加 reranking、answer evaluation、显式受控的 agent layers、
更丰富的 source inspection、web review surfaces，以及更完善的 developer
ergonomics。这些是未来方向，不是当前 v0.2 功能。

更多上下文：

- [docs/ARCHITECTURE.zh-CN.md](docs/ARCHITECTURE.zh-CN.md)
- [docs/RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md)
- [docs/PROJECT_WALKTHROUGH.zh-CN.md](docs/PROJECT_WALKTHROUGH.zh-CN.md)
- [docs/V0_1_SCOPE.zh-CN.md](docs/V0_1_SCOPE.zh-CN.md)
- [docs/TUI_COMMAND_SHELL_DESIGN.zh-CN.md](docs/TUI_COMMAND_SHELL_DESIGN.zh-CN.md)
- [docs/DEMO_SCRIPT.zh-CN.md](docs/DEMO_SCRIPT.zh-CN.md)
- [docs/RELEASE_NOTES_V0_2.md](docs/RELEASE_NOTES_V0_2.md)
- [docs/RELEASE_NOTES_V0_1.zh-CN.md](docs/RELEASE_NOTES_V0_1.zh-CN.md)
- [docs/PORTFOLIO_SUMMARY.zh-CN.md](docs/PORTFOLIO_SUMMARY.zh-CN.md)
