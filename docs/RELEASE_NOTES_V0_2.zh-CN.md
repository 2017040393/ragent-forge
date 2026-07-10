# RAGentForge v0.2 Release Notes

> 语言: [English](RELEASE_NOTES_V0_2.md) | 中文

## Summary

RAGentForge v0.2 把项目从一个本地可检查的 RAG demo，推进为 retrieval quality
engineering foundation。它加入了 BM25、span-grounded eval generation、
evidence-to-current-chunk mapping、更丰富的 retrieval metrics、持久化 eval
reports、deterministic failure analysis、retrieval compare，以及更成熟的
command-first TUI inspection workflow。

这个版本继续保持 local-first 和 explicit：生成状态仍保存在 `.ragent` 下，
retrieval modes 通过命令显式选择，eval artifacts 是可检查、可提交、可比较的
JSON/JSONL/Markdown 文件。

## Why It Matters

RAG quality 工作不能只看一条 answer transcript。团队需要知道检索到了哪些来源、
预期 evidence 是否被命中、组装了多少 context、retrieval 有多慢，以及 miss 是由
chunking、retrieval mode、ranking 还是 dataset design 引起的。

v0.2 让这个流程可重复。Eval cases 可以指向稳定的 source evidence，而不是固定的
chunk ids。评估时，RAGentForge 会把这些 evidence spans 映射到当前 chunk store，
所以当 chunk size、chunk overlap、ingestion 或 retrieval strategy 改变时，同一份
dataset 仍然有用。

## Added

- Markdown/TXT/PDF structured ingestion，通过统一的
  `DocumentBlock[] -> BlockChunker -> DocumentChunk[]` pipeline。
- BM25 retrieval：不需要 embeddings 的更强 sparse baseline。
- Hybrid retrieval：使用 reciprocal rank fusion 融合 BM25 和 semantic results。
- 从稳定 source evidence 生成 span-grounded synthetic eval cases。
- Retrieval eval 中的 evidence-to-current-chunk mapping。
- Retrieval eval metrics：
  - Hit@k
  - Recall@k
  - MRR
  - retrieval latency
  - retrieved context characters
  - estimated context tokens
- `.ragent/eval/runs/` 下的持久化 retrieval eval run directories：
  - `summary.json`
  - `summary.md`
  - `cases.jsonl`
  - `failures.jsonl`
- 带 `failure_type` 和 `failure_reason` 的 deterministic failure analysis。
- `ragent eval compare`：一个命令比较多个 retrieval modes 和 top-k limits。
- Command-first TUI polish：default hybrid Ask、streaming answer display、
  clean chat transcript badges、focused source/session pickers、source
  navigation、inspector context、visual theme、BM25 mode selection、contextual
  command argument suggestions、queued drafts、actionable worker failures 和
  prompt preview。
- 本地 TUI session workbench：latest-session restore、saved turns 和 sources、
  session picker、recent/pinned/starred/failed/has-sources filters、
  pin/star/search、export、branch、rerun、continue-from-sources、auto title 和
  answer-turn selection。
- 适合 review 和 automation 的本地 JSON/JSONL/Markdown eval artifacts。

## Changed

- `ragent eval retrieval` 仍写入 latest compatibility report，现在也会写入带时间戳的
  reproducible run directory。
- Retrieval eval cases 除旧的 chunk/source expectations 外，也可以使用
  `evidence_spans`。
- Eval reports 现在包含 compact per-case records，不包含完整 chunk text、
  embeddings 或 provider secrets。
- `ragent eval compare` 可以从同一个 JSONL cases file 评估多个 retrieval modes 和
  top-k limits。
- `hybrid` retrieval 现在表示 BM25 plus semantic retrieval。Semantic 和 hybrid
  modes 需要 vector index；lexical 和 BM25 不需要。
- TUI 对 ingest、index、eval 和 config mutation 仍保持 read-only，同时会在
  `.ragent/sessions/` 下写入本地 session artifacts。

## How To Try It

从干净本地 workspace 开始：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
uv run ragent chunks list --workspace .ragent --limit 10
```

运行 sparse retrieval baselines：

```bash
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval bm25 --workspace .ragent
```

在 semantic 或 hybrid retrieval 前构建 vector index：

```bash
uv run ragent index build --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval semantic --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

生成 span-grounded eval cases。先用 `--dry-run` 检查 source evidence，不调用
generation provider：

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --dry-run
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --overwrite
```

如果希望 eval generation 包含 text-based PDFs，添加 `--include-pdf`：

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output examples/eval/synthetic_span_cases.jsonl --questions-per-span 2 --max-cases 20 --include-pdf --overwrite
```

运行 retrieval eval 并比较 modes：

```bash
uv run ragent eval retrieval --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval bm25 --limit 5
uv run ragent eval compare --workspace .ragent --cases examples/eval/synthetic_span_cases.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5
```

检查 command-first TUI：

```bash
uv run ragent tui --workspace .ragent
```

在 TUI 中可以尝试：

```text
/search Agentic RAG
/mode bm25
/ask What does agentic RAG add?
/sources
/source next
/prompt on
/sessions
/sessions failed
/sessions has-sources
/export markdown
/turn last
/exit
```

带截图的本地实测 demo 记录在
[V0_2_DEMO_RESULTS.zh-CN.md](V0_2_DEMO_RESULTS.zh-CN.md)。

## Known Limitations

- Semantic 和 hybrid retrieval 需要已配置的 embedding provider 和已构建的
  vector index。
- Synthetic question generation 需要已配置的 generation provider，除非使用
  `--dry-run`。
- PDF support 面向 text-based PDFs。不包含 OCR、scanned PDFs、image text
  recognition 和 PDF rendering。
- Markdown parsing 有意保持轻量、line-based，不是完整 CommonMark。
- Eval metrics 衡量 retrieval behavior，不衡量最终 answer quality。
- TUI 对 ingest、index、eval 和 config mutation workflows 有意保持 read-only。
  它会持久化本地 sessions 和 exports。

## Deferred Work

- Reranking 和 cross-encoder reranking。
- Query rewriting。
- Agentic multi-step retrieval。
- LLM-as-judge answer grading。
- RAGAS integration。
- OCR 和 scanned PDF support。
- PDF viewing/editing 或 source full-text viewing。
- Web dashboard。
- Short demo recordings 和更广泛的 benchmark-style corpora。

这些项目仍在 v0.2 范围之外。现行 roadmap 把 project memory 和 inspectable
single-pass retrieval quality improvements 放在 v0.3，把受控的 multi-step
retrieval 和 agent workflows 放在 v0.4，并在 v0.5 增加本地 comparison views。
参见 [Roadmap](roadmap.zh-CN.md)。
