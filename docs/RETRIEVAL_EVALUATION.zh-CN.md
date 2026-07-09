# Retrieval Evaluation

> 语言: [English](RETRIEVAL_EVALUATION.md) | 中文

RAGentForge v0.2 把 retrieval evaluation 变成一个本地 quality engineering loop。
目标是在没有 hosted service 或 hidden backend 的情况下，让 retrieval quality 可测量、
可比较、可诊断。

核心思想是 span-grounded evaluation。Eval datasets 不应该和某一种具体 chunking
strategy 强绑定。RAGentForge 可以生成并评估 span-grounded retrieval eval cases。
Evidence spans 会在评估时映射到当前 chunk index，因此即使 chunking strategy 改变，
eval cases 仍然稳定。

## Local Workflow

从仓库根目录运行命令。

### 1. Ingest Documents

```bash
uv run ragent ingest examples/knowledge --workspace .ragent
```

### 2. Optional Vector Index

Semantic 和 hybrid retrieval 需要 vector index。BM25 和 lexical 不需要。

```bash
uv run ragent config init --workspace .ragent
# edit .ragent/config.toml for an embedding provider if needed
uv run ragent index build --workspace .ragent
```

### 3. Generate Span-Based Eval Cases

先 dry run。它会抽取并统计 evidence spans，但不调用模型。

```bash
uv run ragent eval generate \
  --source examples/knowledge \
  --workspace .ragent \
  --output examples/eval/synthetic_span_cases.jsonl \
  --dry-run
```

真正生成需要配置 text generation provider，例如在 `.ragent/config.toml` 中设置
`generation.provider = "openai_responses"`。

```bash
uv run ragent eval generate \
  --source examples/knowledge \
  --workspace .ragent \
  --output examples/eval/synthetic_span_cases.jsonl \
  --questions-per-span 2 \
  --max-cases 20 \
  --overwrite
```

如果希望从 text-based PDFs 生成 cases，添加 `--include-pdf`。

### 4. Run Retrieval Eval

```bash
uv run ragent eval retrieval \
  --workspace .ragent \
  --cases examples/eval/synthetic_span_cases.jsonl \
  --retrieval bm25 \
  --limit 5
```

### 5. Compare Retrieval Modes

```bash
uv run ragent eval compare \
  --workspace .ragent \
  --cases examples/eval/synthetic_span_cases.jsonl \
  --retrieval lexical,bm25,semantic,hybrid \
  --limit 1,3,5
```

如果在 vector index 不存在时请求 semantic 或 hybrid run，compare report 会把该 run
记录为 failed，并继续执行，除非使用 `--fail-fast`。

## Retrieval Modes

- `lexical`：简单 token-overlap baseline。
- `bm25`：使用 BM25 scoring 的更强 lexical baseline，不需要 vector index。
- `semantic`：embedding-based vector retrieval，需要
  `uv run ragent index build --workspace .ragent`。
- `hybrid`：BM25 和 semantic retrieval 的 Reciprocal-rank-fusion style 组合，
  需要和 semantic retrieval 相同的 vector index。

Textual Shell TUI 在 retrieval workflows 上仍然是 command-first 和
read-oriented。它默认使用 `hybrid`，并且 `/mode` 命令支持 `lexical`、`bm25`、
`semantic` 和 `hybrid`。

## Evaluation Artifacts

Retrieval eval 会写 compatibility report 和 reproducible run directory：

```text
.ragent/eval/latest_retrieval_eval.json
.ragent/eval/runs/retrieval-YYYYMMDDTHHMMSSZ/
  summary.json
  summary.md
  cases.jsonl
  failures.jsonl
```

- `summary.json`：完整 machine-readable retrieval eval report。
- `summary.md`：human-readable run summary，包含 metrics、report paths 和
  failure breakdown。
- `cases.jsonl`：compact evaluated cases，不包含完整 retrieved chunk text。
- `failures.jsonl`：只包含 failed cases，并带有 `failure_type` 和
  `failure_reason`。

Retrieval compare 会写：

```text
.ragent/eval/latest_retrieval_compare.json
```

`latest_retrieval_compare.json` 汇总每个 requested retrieval mode 和 top-k run，
包括 metrics、status、run paths 和 failures。

## Retrieval Metrics

- `hit@k`：case 在 top-k retrieved results 内是否有任何 matching expected chunk
  或 expected source。
- `recall@k`：top-k 内命中的 expected chunks 比例。如果 case 没有 expected chunks，
  recall 为 `0.0`。
- `mrr`：第一个 matching result 的 mean reciprocal rank。
- `avg_retrieval_latency_ms`：retrieval search 内部耗时的平均值。
- `avg_retrieved_context_chars`：平均 retrieved context size，单位为字符。
- `avg_estimated_context_tokens`：基于字符数除以 4 的简单 context cost estimate。

这些 metrics 只衡量 retrieval behavior，不评价 answer quality。

## Failure Analysis

Failure analysis 是 deterministic 的，用于 debugging。它不是 LLM-as-judge。

- `no_result`：没有返回 retrieval results。
- `unmapped_evidence`：evidence spans 无法映射到当前 chunks。
- `missed_source`：retrieved results 不包含任何 expected source path。
- `wrong_section`：expected source 被检索到，但 top-k 中没有 expected chunk。
- `low_rank`：expected chunks 没有出现在评估的 top-k results 内。
- `unknown`：没有 deterministic failure heuristic 匹配。

如果要检查 individual failures，使用 `failures.jsonl`。如果要观察 recurring failure
modes，使用 `summary.md` 或 `latest_retrieval_compare.json` 中的 failure breakdown。

## Example Compare Table

这个表格是说明性示例，不是 checked-in benchmark result。

```text
mode      k   status   hit@k   recall@k   mrr     avg_latency_ms   failures
lexical   5   success  0.5000  0.4200     0.3900  3.2000           4
bm25      5   success  0.6500  0.5700     0.5100  4.1000           3
semantic  5   success  0.7000  0.6200     0.5600  18.3000          2
hybrid    5   success  0.7800  0.6900     0.6300  22.5000          1
```

## Demo Script

在 interview 或 project walkthrough 中可以使用这个叙事：

1. Ingest 一个小型 local knowledge base。
2. 从 source documents 生成 span-grounded eval cases。
3. 不依赖 embeddings 运行 BM25 retrieval eval。
4. 比较 lexical、BM25、semantic 和 hybrid retrieval。
5. 打开 `failures.jsonl` 并解释 failure types。
6. 解释为什么 evidence spans 让 eval cases 独立于 chunking。

## v0.2 Non-Goals

RAGentForge v0.2 有意不包含：

- Reranking。
- Query rewriting。
- Agentic multi-step retrieval。
- LLM-as-judge answer grading。
- RAGAS integration。
- Web dashboard。

这些是未来方向，不是当前能力。
