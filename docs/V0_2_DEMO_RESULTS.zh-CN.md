# RAGentForge v0.2 Demo Results

> 语言: [English](V0_2_DEMO_RESULTS.md) | 中文

本页记录一次真实的本地 v0.2 demo run。下面的数字来自指定 commit 上的本地 workspace；
它们不是 benchmark claims。只要 corpus、chunking、provider 或 retrieval code 发生变化，
这些数字就应该刷新。

Screenshots 来自真实 command output，绝对本地路径已规范化为 `<repo>`。TUI screenshot
由运行中的 Textual app 导出。

## Environment

| Field | Value |
|---|---|
| Date | `2026-07-07` |
| Git commit | `dc42972` |
| Python | `3.12.11` |
| OS | `Microsoft Windows NT 10.0.26200.0` |
| Workspace | `.ragent` |
| Corpus | `examples/knowledge` |
| Corpus contents | 2 Markdown files, 1 text-based PDF |
| Generation provider | `openai_responses`, model `gpt-5.5` |
| Embedding provider | `openai_embeddings`, model `Qwen/Qwen3-Embedding-0.6B` |

当前本地 corpus 说明：这次 run 覆盖了 `examples/knowledge` 中的 Markdown 和 PDF 文件。
TXT ingestion 是当前实现的一部分，但这次本地 corpus 没有 `.txt` 文件。

## Screenshots

| Screenshot | What It Shows |
|---|---|
| ![v0.2 ingest/status/chunks](assets/v0_2/v0_2_ingest_status_chunks.png) | 混合本地 corpus ingest、status 和 chunk listing，包含 PDF page labels。 |
| ![v0.2 retrieval modes](assets/v0_2/v0_2_retrieval_modes.png) | lexical、BM25、semantic 和 hybrid retrieval 的 top result comparison。 |
| ![v0.2 eval generation](assets/v0_2/v0_2_eval_generation.png) | Span-grounded dry run 以及真实 synthetic eval generation。 |
| ![v0.2 retrieval eval](assets/v0_2/v0_2_retrieval_eval_bm25.png) | BM25@5 retrieval evaluation 和持久化 report paths。 |
| ![v0.2 retrieval compare](assets/v0_2/v0_2_retrieval_compare.png) | 跨 modes 和 top-k limits 的 retrieval compare。 |
| ![v0.2 failure analysis](assets/v0_2/v0_2_failure_analysis.png) | Compare runs 产生的 deterministic failure types。 |
| ![v0.2 TUI shell](assets/v0_2/v0_2_tui_shell.svg) | Command-first TUI，BM25 search、source inspection 和 prompt preview 已启用。 |

## 1. Prepare Workspace

Commands：

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
uv run ragent status --workspace .ragent
uv run ragent chunks list --workspace .ragent --limit 10
```

Observed output summary：

```text
Ingest complete
Documents: 3
Chunks: 796
Skipped files: 0
Chunk size: 1000
Chunk overlap: 0
Workspace: .ragent
Status: ready
```

Chunk listing 确认：

- 来自 `agentic_rag.md` 的 Markdown chunks。
- 来自 `rag_basics.md` 的 Markdown chunks。
- 来自 `High-Dimensional Probability_ An Introduction wit.pdf` 的 PDF chunks。
- PDF chunks 显示 page-aware source labels，例如 `p.2`、`p.3` 和 `p.4`。

## 2. Retrieval Mode Smoke Test

Commands：

```bash
uv run ragent search "What is Agentic RAG?" --retrieval lexical --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval bm25 --workspace .ragent
uv run ragent index build --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval semantic --workspace .ragent
uv run ragent search "What is Agentic RAG?" --retrieval hybrid --workspace .ragent
```

Top result by mode：

| Mode | Top Source | Top Chunk | Score |
|---|---|---|---:|
| lexical | `High-Dimensional Probability_ An Introduction wit.pdf p.91` | `chunk-0230` | `9` |
| bm25 | `agentic_rag.md` | `chunk-0000` | `22.9449` |
| semantic | `agentic_rag.md` | `chunk-0000` | `0.835802` |
| hybrid | `agentic_rag.md` | `chunk-0000` | `0.0327869` |

Interpretation：

- Lexical token overlap 很容易被更大的 PDF corpus 干扰。
- BM25、semantic 和 hybrid 都把预期的 `agentic_rag.md` chunk 放在第一位。
- Hybrid 结合 BM25 和 semantic signals；用户不需要在 query time 手动选择单一 sparse
  或 dense retriever。

## 3. Span-Grounded Eval Generation

Commands：

```bash
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output .ragent/eval/v0_2_demo_generated_cases.jsonl --questions-per-span 2 --max-cases 6 --include-pdf --dry-run
uv run ragent eval generate --source examples/knowledge --workspace .ragent --output .ragent/eval/v0_2_demo_generated_cases.jsonl --questions-per-span 2 --max-cases 6 --include-pdf --overwrite
```

Observed generation output：

| Field | Value |
|---|---:|
| Evidence spans extracted | `300` |
| include_pdf | `True` |
| questions_per_span | `2` |
| max_cases | `6` |
| Cases generated | `6` |
| Spans skipped | `0` |
| Error count | `0` |

Generated case composition：

| Field | Value |
|---|---|
| Output file | `.ragent/eval/v0_2_demo_generated_cases.jsonl` |
| Case IDs | `synthetic-span-000001` through `synthetic-span-000006` |
| Question types | 5 factual, 1 comparison |
| Difficulty | 5 easy, 1 medium |
| Evidence media | 4 PDF evidence spans, 2 Markdown evidence spans |
| Generation method | `llm_synthetic_span_v0.2` |

Generated questions：

| Case | Question |
|---|---|
| `synthetic-span-000001` | What does Agentic RAG add on top of basic retrieval and generation? |
| `synthetic-span-000002` | How is an agentic RAG workflow different from a simple one-shot RAG workflow? |
| `synthetic-span-000003` | Who wrote the high-dimensional probability book mentioned in the evidence? |
| `synthetic-span-000004` | What background does the text assume before introducing high-dimensional probability? |
| `synthetic-span-000005` | Who is the author of the book High-Dimensional Probability? |
| `synthetic-span-000006` | What kinds of students is this book described as useful for in a graduate course? |

## 4. Retrieval Eval

Command：

```bash
uv run ragent eval retrieval --workspace .ragent --cases .ragent/eval/v0_2_demo_generated_cases.jsonl --retrieval bm25 --limit 5
```

Report artifacts：

| Artifact | Path |
|---|---|
| Compatibility report | `.ragent/eval/retrieval_eval_20260707T100034Z-001.json` |
| Run directory | `.ragent/eval/runs/retrieval-20260707T100034Z-001` |
| Summary JSON | `.ragent/eval/runs/retrieval-20260707T100034Z-001/summary.json` |
| Summary Markdown | `.ragent/eval/runs/retrieval-20260707T100034Z-001/summary.md` |
| Cases JSONL | `.ragent/eval/runs/retrieval-20260707T100034Z-001/cases.jsonl` |
| Failures JSONL | `.ragent/eval/runs/retrieval-20260707T100034Z-001/failures.jsonl` |

BM25@5 metrics：

| Metric | Value |
|---|---:|
| cases | `6` |
| passed | `6` |
| failed | `0` |
| hit@1 | `0.5000` |
| hit@3 | `1.0000` |
| hit@5 | `1.0000` |
| hit@k | `1.0000` |
| recall@k | `0.5833` |
| mrr | `0.7500` |
| avg_retrieval_latency_ms | `67.9598` |
| avg_retrieved_count | `5.0000` |
| avg_retrieved_context_chars | `4117.1667` |
| avg_estimated_context_tokens | `1029.5000` |

Failure analysis：

| Failure Type | Count | Notes |
|---|---:|---|
| none | `0` | BM25@5 的 `failures.jsonl` 为空。 |

## 5. Retrieval Compare

Command：

```bash
uv run ragent eval compare --workspace .ragent --cases .ragent/eval/v0_2_demo_generated_cases.jsonl --retrieval lexical,bm25,semantic,hybrid --limit 1,3,5
```

Compare output：

| Retrieval | k | Status | Hit@k | Recall@k | MRR | Avg Latency ms | Failures |
|---|---:|---|---:|---:|---:|---:|---:|
| lexical | 1 | success | `0.1667` | `0.0555` | `0.1667` | `48.1661` | `5` |
| lexical | 3 | success | `0.1667` | `0.0555` | `0.1667` | `53.2369` | `5` |
| lexical | 5 | success | `0.1667` | `0.0555` | `0.1667` | `46.0975` | `5` |
| bm25 | 1 | success | `0.5000` | `0.2639` | `0.5000` | `62.9411` | `3` |
| bm25 | 3 | success | `1.0000` | `0.5278` | `0.7500` | `68.6018` | `0` |
| bm25 | 5 | success | `1.0000` | `0.5833` | `0.7500` | `63.1916` | `0` |
| semantic | 1 | success | `0.5000` | `0.3750` | `0.5000` | `2152.1643` | `3` |
| semantic | 3 | success | `0.8333` | `0.4722` | `0.6389` | `1385.7815` | `1` |
| semantic | 5 | success | `1.0000` | `0.5833` | `0.6722` | `1373.3224` | `0` |
| hybrid | 1 | success | `0.6667` | `0.4305` | `0.6667` | `1271.1520` | `2` |
| hybrid | 3 | success | `1.0000` | `0.5833` | `0.8056` | `1342.3165` | `0` |
| hybrid | 5 | success | `1.0000` | `0.6250` | `0.8056` | `1226.3193` | `0` |

Compare artifacts：

| Artifact | Path |
|---|---|
| Compare report | `.ragent/eval/retrieval_compare_20260707T100128Z.json` |
| Individual run directories | `.ragent/eval/runs/` |

Failure breakdown from compare runs：

| Retrieval | k | Failures | Failure Breakdown |
|---|---:|---:|---|
| lexical | 1 | `5` | `missed_source: 2`, `wrong_section: 3` |
| lexical | 3 | `5` | `missed_source: 2`, `wrong_section: 3` |
| lexical | 5 | `5` | `missed_source: 2`, `wrong_section: 3` |
| bm25 | 1 | `3` | `missed_source: 1`, `wrong_section: 2` |
| semantic | 1 | `3` | `wrong_section: 3` |
| semantic | 3 | `1` | `wrong_section: 1` |
| hybrid | 1 | `2` | `wrong_section: 2` |

Interpretation：

- 在这次本地 run 中，`hybrid@5` 的 recall@k (`0.6250`) 和 MRR (`0.8056`) 最高，
  同时 6 个 cases 全部通过。
- `bm25@3` 和 `bm25@5` 也通过了全部 6 个 cases，并且比 semantic/hybrid 快得多，
  因为它们没有调用 embedding search path。
- Lexical 仍是有用 baseline，但在这个 mixed corpus 上表现吃力，因为大型 PDF 贡献了很多
  high-overlap distractor chunks。

## 6. TUI Smoke Check

TUI screenshot 是通过 Textual 的 `save_screenshot` API 捕获的，当时以 `.ragent`
启动 `RagentForgeApp` 并提交了这些 commands：

```text
/mode bm25
/search What is Agentic RAG?
/sources
/source 1
/prompt on
```

Observed checks：

| Check | Result | Notes |
|---|---|---|
| TUI launches | pass | Textual test harness 打开了 `RagentForgeApp`。 |
| BM25 mode is selectable | pass | Status 变为 `mode: bm25`。 |
| Search completes | pass | BM25 search 从 local chunks 返回 sources。 |
| Sources are navigable | pass | `/source 1` 选中了第一个 source。 |
| Inspector shows selected source | pass | Inspector 显示 selected source details。 |
| Prompt preview toggles | pass | `/prompt on` 启用了 shell state 中的 prompt preview。 |

## Final Notes

- 这次本地 run 中 quality 最好的 mode：`hybrid@5`，依据 recall@k 和 MRR。
- 这次本地 run 中最快的 sparse mode：`bm25@5`，依据全部 cases 通过且 latency 远低于
  semantic/hybrid。
- 最有用的 failure types：`missed_source` 和 `wrong_section`。
- Dataset cleanup note：generated dataset 很小（`6` cases），适合 demo，不是 benchmark。
- Retrieval improvement note：lexical baseline 在这个 corpus 上较弱，因为 exact token
  overlap 被大型 PDF 干扰；对于 mixed local corpora，BM25 是更好的 sparse default。
