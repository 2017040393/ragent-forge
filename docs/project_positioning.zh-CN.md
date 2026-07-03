# Project Positioning

> 语言: [English](project_positioning.md) | 中文

RAGentForge 是一个本地优先的 TUI 应用，用于在个人知识库上构建可检查的
Agentic RAG workflows。第一批用户是希望从终端处理自己的 Markdown 和 TXT
笔记、项目文档、学习材料和面试准备文档的开发者。

RAGentForge 不是 no-code platform、enterprise knowledge base、desktop app、
cloud service 或 autonomous agent framework。早期版本应保持小、清晰，并且
有助于学习 RAG systems 的行为。

Local-first 很重要，因为个人知识通常保存在开发者机器上，并且可能包含私人
笔记、草稿和项目上下文。默认 workflow 不应要求托管服务、账号或隐藏网络调用。

TUI-first 让应用贴近开发者 workflow。终端界面迭代快，易于本地运行，也很适合
可检查的 text-heavy workflows。

Inspectability 优先于 autonomy。项目在尝试更自主的行为前，用户应能看到加载
了什么、文本如何被切块、检索到了哪些 sources，以及什么 trace 导向了答案。

Python 是早期版本的主要实现语言，因为 AI 和 RAG 生态变化很快，Python 能让
实验保持轻量。

Rust 被有意推迟。未来它可能对 performance-critical modules 有用，但项目应等
到有测量出的 bottlenecks 后，再增加 native extensions 或 mixed-language build。
