# Structured Ingestion Demo Workflow

> 语言: [English](STRUCTURED_INGESTION_DEMO.md) | 中文

这个 workflow 展示 structured ingestion milestone。它展示 Markdown、TXT 和 PDF 文件
如何流经同一种 ingestion shape：

```text
Document + DocumentBlock[] -> BlockChunker -> DocumentChunk[]
```

Demo 使用隔离的临时 corpus 和 workspace，因此不会修改你平时使用的 `.ragent`
workspace。

## Demo Goal

证明当前实现可以：

- 一起 ingest Markdown、TXT 和 PDF 文件；
- 保留 Markdown section metadata；
- 保留 TXT character ranges；
- 保留 PDF page 和 table metadata；
- 使用现有 CLI commands 对生成的 chunks 执行 search 和 ask。

## Requirements

从仓库根目录运行：

```bash
uv sync --extra dev
```

`dev` extra 包含 `reportlab`，demo 用它创建一个很小的 PDF。

## 1. Create a Mixed Demo Corpus

### Bash, WSL, Git Bash, or macOS/Linux shell

```bash
DEMO_ROOT="$(mktemp -d)"
DEMO_CORPUS="$DEMO_ROOT/corpus"
DEMO_WORKSPACE="$DEMO_ROOT/workspace"
export DEMO_CORPUS DEMO_WORKSPACE

uv run --extra dev python - <<'PY'
import os
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

corpus = Path(os.environ["DEMO_CORPUS"])
corpus.mkdir(parents=True, exist_ok=True)

(corpus / "rag_notes.md").write_text(
    "# RAG Basics\n\n"
    "Retrieval augmented generation combines search with generation.\n\n"
    "## Hybrid Retrieval\n\n"
    "| Method | Strength |\n"
    "|---|---|\n"
    "| lexical | exact terms |\n"
    "| semantic | meaning |\n\n"
    "Hybrid retrieval can combine both signals.\n",
    encoding="utf-8",
)

(corpus / "plain_notes.txt").write_text(
    "Plain text notes keep character ranges.\n\n"
    "A second paragraph becomes another paragraph block.\n",
    encoding="utf-8",
)

pdf_path = corpus / "pdf_notes.pdf"
pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
pdf.drawString(72, 740, "PDF Notes")
pdf.drawString(72, 720, "PDF ingestion keeps page metadata.")
pdf.drawString(72, 700, "Table 1: Retrieval Signals")
pdf.drawString(72, 680, "Method | Strength")
pdf.drawString(72, 660, "lexical | exact terms")
pdf.drawString(72, 640, "semantic | meaning")
pdf.save()

print(corpus)
PY
```

### Windows PowerShell

```powershell
$DemoRoot = Join-Path $env:TEMP ("ragent-structured-demo-" + [guid]::NewGuid())
$env:DEMO_CORPUS = Join-Path $DemoRoot "corpus"
$env:DEMO_WORKSPACE = Join-Path $DemoRoot "workspace"

@'
import os
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

corpus = Path(os.environ["DEMO_CORPUS"])
corpus.mkdir(parents=True, exist_ok=True)

(corpus / "rag_notes.md").write_text(
    "# RAG Basics\n\n"
    "Retrieval augmented generation combines search with generation.\n\n"
    "## Hybrid Retrieval\n\n"
    "| Method | Strength |\n"
    "|---|---|\n"
    "| lexical | exact terms |\n"
    "| semantic | meaning |\n\n"
    "Hybrid retrieval can combine both signals.\n",
    encoding="utf-8",
)

(corpus / "plain_notes.txt").write_text(
    "Plain text notes keep character ranges.\n\n"
    "A second paragraph becomes another paragraph block.\n",
    encoding="utf-8",
)

pdf_path = corpus / "pdf_notes.pdf"
pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
pdf.drawString(72, 740, "PDF Notes")
pdf.drawString(72, 720, "PDF ingestion keeps page metadata.")
pdf.drawString(72, 700, "Table 1: Retrieval Signals")
pdf.drawString(72, 680, "Method | Strength")
pdf.drawString(72, 660, "lexical | exact terms")
pdf.drawString(72, 640, "semantic | meaning")
pdf.save()

print(corpus)
'@ | uv run --extra dev python -
```

## 2. Ingest the Mixed Corpus

```bash
uv run ragent ingest "$DEMO_CORPUS" --workspace "$DEMO_WORKSPACE"
uv run ragent status --workspace "$DEMO_WORKSPACE"
uv run ragent chunks list --workspace "$DEMO_WORKSPACE" --limit 20
```

PowerShell 使用同样的命令，只是变量写成 `$env:`：

```powershell
uv run ragent ingest $env:DEMO_CORPUS --workspace $env:DEMO_WORKSPACE
uv run ragent status --workspace $env:DEMO_WORKSPACE
uv run ragent chunks list --workspace $env:DEMO_WORKSPACE --limit 20
```

预期讲法：

- Markdown chunks 显示 `rag_notes.md` 和 character ranges。
- TXT chunks 显示 `plain_notes.txt` 和 character ranges。
- PDF chunks 在适用时显示带 page-based ranges 的 `pdf_notes.pdf`。

## 3. Inspect Structured Metadata

打印每个 generated chunk 的 media type、block types、character range、page range 和
section metadata。

### Bash, WSL, Git Bash, or macOS/Linux shell

```bash
uv run python - <<'PY'
import json
import os
from pathlib import Path

chunks_path = Path(os.environ["DEMO_WORKSPACE"]) / "chunks" / "chunks.jsonl"
for line in chunks_path.read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    metadata = record["metadata"]
    print(record["source_path"])
    print("  media_type:", metadata.get("media_type"))
    print("  block_types:", metadata.get("block_types"))
    print("  char_range:", record.get("start_char"), record.get("end_char"))
    print("  page_range:", metadata.get("page_start"), metadata.get("page_end"))
    print("  section:", metadata.get("section_title"))
    print("  heading_path:", metadata.get("heading_path"))
PY
```

### Windows PowerShell

```powershell
@'
import json
import os
from pathlib import Path

chunks_path = Path(os.environ["DEMO_WORKSPACE"]) / "chunks" / "chunks.jsonl"
for line in chunks_path.read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    metadata = record["metadata"]
    print(record["source_path"])
    print("  media_type:", metadata.get("media_type"))
    print("  block_types:", metadata.get("block_types"))
    print("  char_range:", record.get("start_char"), record.get("end_char"))
    print("  page_range:", metadata.get("page_start"), metadata.get("page_end"))
    print("  section:", metadata.get("section_title"))
    print("  heading_path:", metadata.get("heading_path"))
'@ | uv run python -
```

可以指出：

- Markdown 有 `media_type: text/markdown`。
- TXT 有 `media_type: text/plain`。
- PDF 有 `media_type: application/pdf`。
- Markdown/TXT chunks 有 `start_char` 和 `end_char`。
- PDF chunks 不需要 character ranges；它们携带 page metadata。
- Markdown chunks 可以携带 `section_title` 和 `heading_path`。

## 4. Search and Ask

```bash
uv run ragent search "Hybrid Retrieval" --retrieval lexical --workspace "$DEMO_WORKSPACE"
uv run ragent ask "What does hybrid retrieval combine?" --retrieval lexical --workspace "$DEMO_WORKSPACE"
```

PowerShell：

```powershell
uv run ragent search "Hybrid Retrieval" --retrieval lexical --workspace $env:DEMO_WORKSPACE
uv run ragent ask "What does hybrid retrieval combine?" --retrieval lexical --workspace $env:DEMO_WORKSPACE
```

预期讲法：

- Search 和 Ask 仍然使用同一套 public commands。
- Ingestion 之后 retrieval 是 chunk-based 且 format-agnostic。
- Source labels 保持 compact，同时 inspectors 可以显示更丰富的 metadata。

## 5. Optional TUI Inspection

如果想让 TUI 读取这个 demo workspace，显式传入 demo workspace path：

```bash
uv run ragent tui --workspace "$DEMO_WORKSPACE"
```

然后使用：

```text
/search Hybrid Retrieval
/sources
/source next
/trace
/exit
```

Selected-source Inspector 应该保留 PDF metadata 可见，并显示 concise Markdown/TXT
metadata，例如 type、section、heading path 和 block type。

## 6. Cleanup

Bash：

```bash
rm -rf "$DEMO_ROOT"
```

PowerShell：

```powershell
Remove-Item -Recurse -Force (Split-Path $env:DEMO_CORPUS)
```

## Demo Summary

当前实现保持用户可见 command surface 稳定，同时统一 ingestion internals。Markdown、TXT
和 PDF 都会先变成 structured blocks，再进行 chunking；retrieval 继续在 ordinary chunks
上工作。
