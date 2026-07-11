# 统一检索入口与 Typed Sources

- 日期：2026-07-11
- 状态：accepted-direction
- 相关版本：v0.3
- 相关文档：[v0.3 roadmap](../docs/roadmap.zh-CN.md#v03-retrieval-quality-and-efficiency-engineering)

## 触发背景

v0.3 需要同时提升检索效率、召回率和精确度，并逐步引入 project memory。讨论中
需要明确 document 与 memory 是否应成为不同的检索系统，以及最终产品是否可能
采用 memory-first 形态。

## 当前想法

RAGentForge 使用同一个 retrieval 入口，但保留每条记录的 source type、provenance
和 lifecycle semantics。

统一入口可以覆盖：

- `document_evidence`
- `project_fact`
- `user_note`
- `session_memory`

Project memory 可以成为主要的用户侧 knowledge surface，但不能抹去 document
evidence 的文件、页码、span 和其他可审计来源信息。

## 为什么重要

- 调用方不需要理解多个检索系统，pipeline、trace 和 evaluation 可以复用。
- Document evidence 仍可作为原始证据，支持 citation、更新检测和冲突判断。
- Memory 可以表达项目决策、用户偏好和对话中形成的事实，而不伪装成原始文档。
- 将来可以调整物理存储或索引方式，而不改变上层 retrieval contract。

## 对评测的影响

- v0.3 初期继续复用 v0.2 retrieval eval 作为主 baseline。
- Project memory 出现后，再增加写入、更新、遗忘、过期和 provenance correctness
  cases。
- 只有当产品明确支持跨来源检索、冲突处理或 combined context 时，才增加
  mixed-source benchmark。

## 未决问题

- 统一 retrieval result 最少需要保留哪些 typed metadata？
- 不同 source type 的 authority、时效性和冲突优先级如何表达？
- Memory-first 用户体验从哪个 v0.3 milestone 开始成为默认路径？
- 哪些 source-specific 指标值得进入 release gate？

## 下一步验证

先冻结 v0.2 baseline 和 v0.3 量化标准，再根据瓶颈选择检索技术，不在本条笔记中
提前确定 query rewriting、reranking、索引或存储方案。

