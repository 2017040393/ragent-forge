# Structured Ingestion Demo Workflow

This workflow demonstrates the structured ingestion milestone. It
shows Markdown, TXT, and PDF files flowing through the same ingestion shape:

```text
Document + DocumentBlock[] -> BlockChunker -> DocumentChunk[]
```

The demo uses an isolated temporary corpus and workspace so it does not modify
your normal `.ragent` workspace.

## Demo Goal

Prove that the current implementation can:

- ingest Markdown, TXT, and PDF files together;
- preserve Markdown section metadata;
- preserve TXT character ranges;
- preserve PDF page and table metadata;
- search and ask over the resulting chunks with the existing CLI commands.

## Requirements

Run from the repository root:

```bash
uv sync --extra dev
```

The `dev` extra includes `reportlab`, which the demo uses to create a tiny PDF.

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

PowerShell uses the same commands with `$env:` variables:

```powershell
uv run ragent ingest $env:DEMO_CORPUS --workspace $env:DEMO_WORKSPACE
uv run ragent status --workspace $env:DEMO_WORKSPACE
uv run ragent chunks list --workspace $env:DEMO_WORKSPACE --limit 20
```

Expected story:

- Markdown chunks show `rag_notes.md` and character ranges.
- TXT chunks show `plain_notes.txt` and character ranges.
- PDF chunks show `pdf_notes.pdf` with page-based ranges when applicable.

## 3. Inspect Structured Metadata

Print the media type, block types, character range, page range, and section
metadata for each generated chunk.

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

What to point out:

- Markdown has `media_type: text/markdown`.
- TXT has `media_type: text/plain`.
- PDF has `media_type: application/pdf`.
- Markdown/TXT chunks have `start_char` and `end_char`.
- PDF chunks do not need character ranges; they carry page metadata instead.
- Markdown chunks can carry `section_title` and `heading_path`.

## 4. Search and Ask

```bash
uv run ragent search "Hybrid Retrieval" --retrieval lexical --workspace "$DEMO_WORKSPACE"
uv run ragent ask "What does hybrid retrieval combine?" --retrieval lexical --workspace "$DEMO_WORKSPACE"
```

PowerShell:

```powershell
uv run ragent search "Hybrid Retrieval" --retrieval lexical --workspace $env:DEMO_WORKSPACE
uv run ragent ask "What does hybrid retrieval combine?" --retrieval lexical --workspace $env:DEMO_WORKSPACE
```

Expected story:

- Search and Ask still use the same public commands.
- Retrieval is chunk-based and format-agnostic after ingestion.
- Source labels stay compact, while inspectors can show richer metadata.

## 5. Optional TUI Inspection

If you want the TUI to read this demo workspace, launch it from a directory where
`.ragent` points to the demo workspace, or rerun ingest into the default
workspace:

```bash
uv run ragent ingest "$DEMO_CORPUS" --workspace .ragent
uv run ragent tui
```

Then use:

```text
/search Hybrid Retrieval
/sources
/source next
/trace
/exit
```

The selected-source Inspector should keep PDF metadata visible and show concise
Markdown/TXT metadata such as type, section, heading path, and block type.

## 6. Cleanup

Bash:

```bash
rm -rf "$DEMO_ROOT"
```

PowerShell:

```powershell
Remove-Item -Recurse -Force (Split-Path $env:DEMO_CORPUS)
```

## Demo Summary

The current implementation keeps the user-facing command surface stable while
unifying the ingestion internals. Markdown, TXT, and PDF all become structured
blocks before chunking, and retrieval continues to work over ordinary chunks.
