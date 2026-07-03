# Roadmap

> 语言: [English](roadmap.md) | 中文

## v0.1: Local TUI + Inspectable RAG

目标：

- 加载本地 Markdown/TXT 文件。
- 确定性切分文档。
- 支持 lexical、semantic 和 hybrid retrieval。
- 支持带来源的 Ask 和可选 generation。
- 增加本地 traces 和 retrieval evaluation。
- 在 command-first TUI 中显示 sources、traces、settings 和 source inspection。

非目标：

- 真正的 autonomous agents。
- Cloud sync 或 hosted services。
- PDF ingestion 或复杂文档解析。

## v0.2: Retrieval Quality + Better Source Inspection

目标：

- 在当前 token-overlap baseline 之外改进 lexical retrieval quality。
- 增加 retrieval comparison workflows。
- 让 retrieval scores 和 source selection 更容易检查。
- 改进 trace display、export 和 demo polish。

非目标：

- Enterprise search features。
- Multi-user collaboration。
- Provider-specific lock-in。

## v0.3: Project Memory

目标：

- 增加 workspace-local memory，用于 project facts 和 user-curated notes。
- 保持 memory 可编辑、可审计。
- 区分 retrieved evidence 和 remembered project context。

非目标：

- Hidden long-term memory。
- Cloud profiles。
- 自动导入无关文件。

## v0.4: Minimal Agent Runtime

目标：

- 增加一个小型、受控 runtime，用于显式 multi-step workflows。
- 要求可见的 plans、tool steps 和 trace output。
- 保持 side effects 由用户控制。

非目标：

- Fully autonomous background agents。
- Browser automation。
- Distributed task execution。

## v0.5: Evaluation Dashboard

目标：

- 在本地 test sets 上跟踪 retrieval quality 和 answer quality。
- 为 prompts 和 retrieval settings 增加简单 comparison views。
- 支持 learning-oriented experiments。

非目标：

- Enterprise observability。
- Hosted analytics。
- Complex model evaluation infrastructure。

## v0.6: Open-source Polish

目标：

- 改进 documentation 和 examples。
- 稳定 public interfaces。
- 增加 contributor guidance 和 release workflows。

非目标：

- Large marketplace 或 plugin ecosystem。
- Desktop packaging。
- Production SaaS features。
