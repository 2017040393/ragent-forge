# Ideas and Learning Notes

这个目录用于记录项目中的想法、学习笔记、待验证假设和尚未进入正式设计的方向。
目录内容由 Git 跟踪，因此在提交并推送后，可以在其他电脑上继续整理。

## 使用方式

1. 来不及整理的内容先写入 [inbox.md](inbox.md)。
2. 值得继续研究的内容复制 [_template.md](_template.md)，保存为
   `YYYY-MM-DD-short-topic.md`。
3. 每条笔记尽量记录触发背景、当前判断、依据、未决问题和下一步验证方式。
4. 当想法成为稳定决策或已实现功能时，把结论同步到 `docs/roadmap.md`、
   `docs/ARCHITECTURE.md` 或对应正式文档，并在笔记中链接过去。
5. 用小而清晰的 Git commit 提交笔记；只有已经 commit 并 push 的内容才能可靠地
   跨电脑延续。

## 状态约定

- `seed`: 刚记录，尚未验证。
- `exploring`: 正在收集资料或做实验。
- `accepted-direction`: 已接受为方向，但技术方案尚未确定。
- `implemented`: 已落地，并应链接到实现或正式文档。
- `superseded`: 已被后续想法或决策替代。
- `rejected`: 已明确不采用，并保留原因。

## 边界

- Ideas 不是正式规范；正式行为以代码、测试和 `docs/` 为准。
- 不要记录 API key、token、账号、私人数据或不适合进入公开仓库的信息。
- 引用外部资料时保留链接和访问日期，区分事实、推断和个人判断。
- 不把 `.ragent` 运行数据、生成索引或大体积实验产物提交到此目录。

## 当前条目

- [统一检索入口与 typed sources](2026-07-11-unified-retrieval-typed-sources.md)
- [v0.3 retrieval 研究顺序](2026-07-11-v0-3-retrieval-research-order.md)
- [v0.2 retrieval baseline for v0.3](2026-07-11-v0-2-retrieval-baseline.md)
- [Pre-v0.3 post-architecture retrieval baseline](2026-07-14-pre-v0-3-post-architecture-baseline.md)
- [Retrieval representation screening protocol](2026-07-14-retrieval-screening-protocol.md)
