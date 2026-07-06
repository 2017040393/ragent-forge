import json
from pathlib import Path

from ragent_forge.app.services import evidence_span_service
from ragent_forge.app.services.retrieval_eval_service import RetrievalEvalService
from ragent_forge.cli import build_parser, main


class FakeEmbeddingResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def install_fake_embedding_post(monkeypatch):
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        assert url == "https://api.openai.com/v1/embeddings"
        assert headers["Authorization"] == "Bearer embedding-secret-key"
        assert headers["Content-Type"] == "application/json"

        data = []
        for index, text in enumerate(json["input"]):
            normalized_text = str(text).lower()
            if "retrieval" in normalized_text:
                embedding = [0.0, 1.0]
            elif "agent" in normalized_text:
                embedding = [1.0, 0.0]
            else:
                embedding = [0.5, 0.5]
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": embedding,
                }
            )
        return FakeEmbeddingResponse(
            {
                "object": "list",
                "data": data,
                "model": json["model"],
                "usage": {"total_tokens": len(json["input"])},
            }
        )

    monkeypatch.setattr(
        "ragent_forge.app.services.embedding_service.httpx.post",
        fake_post,
    )
    return calls


def write_embedding_config(workspace_dir: Path) -> None:
    (workspace_dir / "config.toml").write_text(
        (
            "[generation]\n"
            'provider = "null"\n'
            "\n"
            "[embedding]\n"
            'provider = "openai_embeddings"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "text-embedding-3-small"\n'
            'api_key = "embedding-secret-key"\n'
            "timeout_seconds = 60\n"
            "batch_size = 64\n"
        ),
        encoding="utf-8",
    )


def write_generation_config(workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "config.toml").write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = "generation-secret-key"\n'
            "timeout_seconds = 60\n"
            "temperature = 0.2\n"
            "\n"
            "[embedding]\n"
            'provider = "none"\n'
        ),
        encoding="utf-8",
    )


class FakeEvalTextGenerationClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str | None]] = []

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        if not self.responses:
            raise AssertionError("unexpected eval generator call")
        return self.responses.pop(0)


def write_retrieval_eval_cases(
    cases_path: Path,
    records: list[dict[str, object]],
) -> None:
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def write_pdf_chunk_record(workspace_dir: Path) -> None:
    chunks_dir = workspace_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    record = {
        "chunk_id": "/knowledge/paper.pdf::chunk-0000",
        "document_id": "/knowledge/paper.pdf",
        "source_path": "/knowledge/paper.pdf",
        "start_char": None,
        "end_char": None,
        "text": "agent memory on a PDF page",
        "metadata": {
            "source_path": "/knowledge/paper.pdf",
            "media_type": "application/pdf",
            "page_start": 7,
            "page_end": 7,
            "block_types": ["paragraph"],
            "extraction_method": "pdf_structured",
        },
    }
    (chunks_dir / "chunks.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_ingest_help_mentions_pdf() -> None:
    help_text = " ".join(build_parser().format_help().split())

    assert (
        "Ingest local Markdown/TXT/PDF knowledge folders or files."
        in help_text
    )


def test_ingest_command_prints_statistics(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("abcdefghij", encoding="utf-8")
    (knowledge_dir / "notes.txt").write_text("klmnopqrst", encoding="utf-8")
    (knowledge_dir / "skip.bin").write_text("ignored", encoding="utf-8")

    workspace_dir = tmp_path / ".ragent"

    exit_code = main(
        [
            "ingest",
            str(knowledge_dir),
            "--chunk-size",
            "5",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Ingest complete" in captured.out
    assert "Documents: 2" in captured.out
    assert "Chunks: 4" in captured.out
    assert "Skipped files: 1" in captured.out
    assert "Saved chunks to:" in captured.out
    assert "chunks.jsonl" in captured.out
    assert "Saved summary to:" in captured.out
    assert "latest_summary.json" in captured.out
    assert "Saved trace to:" in captured.out
    assert "latest_trace.json" in captured.out
    assert (workspace_dir / "chunks" / "chunks.jsonl").is_file()
    assert (workspace_dir / "ingest" / "latest_summary.json").is_file()
    assert (workspace_dir / "traces" / "latest_trace.json").is_file()

    summary = json.loads(
        (workspace_dir / "ingest" / "latest_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["document_count"] == 2
    assert summary["chunk_count"] == 4
    assert summary["skipped_count"] == 1

    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert latest_trace["operation"] == "ingest"
    assert latest_trace["status"] == "success"
    assert latest_trace["metadata"]["chunk_count"] == 4


def test_ingest_command_reports_errors(tmp_path: Path, capsys) -> None:
    exit_code = main(["ingest", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Ingest failed" in captured.out
    assert "Ingest path not found" in captured.out


def test_status_command_prints_not_initialized_for_missing_workspace(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["status", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Workspace:" in captured.out
    assert "Status: not initialized" in captured.out
    assert "Run `ragent ingest <path>`" in captured.out


def test_status_command_prints_ready_after_ingest(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("abcdefghij", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"

    ingest_exit_code = main(
        [
            "ingest",
            str(knowledge_dir),
            "--chunk-size",
            "5",
            "--workspace",
            str(workspace_dir),
        ]
    )
    assert ingest_exit_code == 0
    capsys.readouterr()

    status_exit_code = main(["status", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert status_exit_code == 0
    assert "Status: ready" in captured.out
    assert "Last ingest source:" in captured.out
    assert "Documents: 1" in captured.out
    assert "Chunks: 2" in captured.out
    assert "Skipped files: 0" in captured.out
    assert "Chunks file:" in captured.out
    assert "Summary file:" in captured.out


def test_status_command_prints_incomplete_workspace(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    (workspace_dir / "chunks").mkdir(parents=True)
    (workspace_dir / "ingest").mkdir(parents=True)

    exit_code = main(["status", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Status: incomplete" in captured.out
    assert "Missing chunks file:" in captured.out
    assert "chunks.jsonl" in captured.out
    assert "Missing summary file:" in captured.out
    assert "latest_summary.json" in captured.out
    assert "Run `ragent ingest <path>`" in captured.out


def test_status_command_reports_corrupt_workspace_json(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    chunks_dir = workspace_dir / "chunks"
    ingest_dir = workspace_dir / "ingest"
    chunks_dir.mkdir(parents=True)
    ingest_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("not-json\n", encoding="utf-8")
    (ingest_dir / "latest_summary.json").write_text("{}", encoding="utf-8")

    exit_code = main(["status", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Status failed" in captured.out
    assert "Invalid JSON in chunks file" in captured.out


def test_config_show_prints_default_when_config_is_missing(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["config", "show", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Config: default" in captured.out
    assert "generation.provider: null" in captured.out
    assert "embedding.provider: none" in captured.out


def test_config_init_writes_default_config(tmp_path: Path, capsys) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["config", "init", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Wrote default config to:" in captured.out
    assert "config.toml" in captured.out
    assert (workspace_dir / "config.toml").read_text(encoding="utf-8") == (
        '[generation]\nprovider = "null"\n'
        "\n"
        '[embedding]\nprovider = "none"\n'
    )


def test_config_init_does_not_overwrite_existing_config(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    workspace_dir.mkdir()
    config_path = workspace_dir / "config.toml"
    config_path.write_text('[generation]\nprovider = "custom"\n', encoding="utf-8")

    exit_code = main(["config", "init", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Config already exists:" in captured.out
    assert config_path.read_text(encoding="utf-8") == (
        '[generation]\nprovider = "custom"\n'
    )


def test_config_init_overwrite_replaces_existing_config(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    workspace_dir.mkdir()
    config_path = workspace_dir / "config.toml"
    config_path.write_text('[generation]\nprovider = "custom"\n', encoding="utf-8")

    exit_code = main(
        ["config", "init", "--overwrite", "--workspace", str(workspace_dir)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Wrote default config to:" in captured.out
    assert config_path.read_text(encoding="utf-8") == (
        '[generation]\nprovider = "null"\n'
        "\n"
        '[embedding]\nprovider = "none"\n'
    )


def test_config_show_reads_valid_config(tmp_path: Path, capsys) -> None:
    workspace_dir = tmp_path / ".ragent"
    workspace_dir.mkdir()
    config_path = workspace_dir / "config.toml"
    config_path.write_text('[generation]\nprovider = "null"\n', encoding="utf-8")

    exit_code = main(["config", "show", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Config: {config_path}" in captured.out
    assert "generation.provider: null" in captured.out
    assert "embedding.provider: none" in captured.out


def test_config_show_reads_openai_responses_config_without_printing_api_key(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    workspace_dir.mkdir()
    config_path = workspace_dir / "config.toml"
    config_path.write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = "super-secret-key"\n'
            "timeout_seconds = 60\n"
            "temperature = 0.2\n"
            'reasoning_effort = "medium"\n'
        ),
        encoding="utf-8",
    )

    exit_code = main(["config", "show", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Config: {config_path}" in captured.out
    assert "generation.provider: openai_responses" in captured.out
    assert "generation.base_url: https://api.openai.com/v1" in captured.out
    assert "generation.model: gpt-4o-mini" in captured.out
    assert "generation.api_key: <hidden>" in captured.out
    assert "generation.timeout_seconds: 60" in captured.out
    assert "generation.temperature: 0.2" in captured.out
    assert "generation.reasoning_effort: medium" in captured.out
    assert "super-secret-key" not in captured.out


def test_config_show_reads_openai_embeddings_config_without_printing_api_key(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    workspace_dir.mkdir()
    config_path = workspace_dir / "config.toml"
    write_embedding_config(workspace_dir)

    exit_code = main(["config", "show", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Config: {config_path}" in captured.out
    assert "embedding.provider: openai_embeddings" in captured.out
    assert "embedding.base_url: https://api.openai.com/v1" in captured.out
    assert "embedding.model: text-embedding-3-small" in captured.out
    assert "embedding.api_key: <hidden>" in captured.out
    assert "embedding.timeout_seconds: 60" in captured.out
    assert "embedding.batch_size: 64" in captured.out
    assert "embedding-secret-key" not in captured.out


def test_config_show_reports_invalid_toml(tmp_path: Path, capsys) -> None:
    workspace_dir = tmp_path / ".ragent"
    workspace_dir.mkdir()
    (workspace_dir / "config.toml").write_text("[generation\n", encoding="utf-8")

    exit_code = main(["config", "show", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Config failed:" in captured.out
    assert "Invalid TOML in config file" in captured.out


def test_config_show_reports_unsupported_provider(tmp_path: Path, capsys) -> None:
    workspace_dir = tmp_path / ".ragent"
    workspace_dir.mkdir()
    (workspace_dir / "config.toml").write_text(
        '[generation]\nprovider = "openai"\n',
        encoding="utf-8",
    )

    exit_code = main(["config", "show", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Config failed:" in captured.out
    assert "Unsupported generation provider: openai" in captured.out


def test_chunks_list_command_prints_chunk_rows_after_ingest(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("abcdefghij", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"

    ingest_exit_code = main(
        [
            "ingest",
            str(knowledge_dir),
            "--chunk-size",
            "5",
            "--workspace",
            str(workspace_dir),
        ]
    )
    assert ingest_exit_code == 0
    capsys.readouterr()

    exit_code = main(["chunks", "list", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Chunk ID" in captured.out
    assert "Source" in captured.out
    assert "Range" in captured.out
    assert "Preview" in captured.out
    assert "rag.md::chunk-0000" in captured.out
    assert "0-5" in captured.out
    assert "abcde" in captured.out


def test_chunks_list_command_formats_pdf_page_range(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    write_pdf_chunk_record(workspace_dir)

    exit_code = main(["chunks", "list", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paper.pdf p.7" in captured.out
    assert "p.7" in captured.out
    assert "None-None" not in captured.out


def test_chunks_list_command_respects_limit_and_prints_summary(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("abcdefghij", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "5",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        ["chunks", "list", "--workspace", str(workspace_dir), "--limit", "1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "rag.md::chunk-0000" in captured.out
    assert "rag.md::chunk-0001" not in captured.out
    assert "Showing 1 of 2 chunks. Use --limit to show more." in captured.out


def test_chunks_show_command_prints_full_chunk_details_and_text(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("abcdefghij", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "5",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    chunks = json.loads(
        (workspace_dir / "chunks" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    exit_code = main(
        ["chunks", "show", chunks["chunk_id"], "--workspace", str(workspace_dir)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Chunk ID: {chunks['chunk_id']}" in captured.out
    assert f"Document ID: {chunks['document_id']}" in captured.out
    assert "Source path:" in captured.out
    assert "Start char: 0" in captured.out
    assert "End char: 5" in captured.out
    assert "Metadata:" in captured.out
    assert "Text:" in captured.out
    assert "abcde" in captured.out


def test_chunks_show_command_prints_not_found_for_missing_chunk(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    chunks_dir = workspace_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("", encoding="utf-8")

    exit_code = main(
        ["chunks", "show", "missing", "--workspace", str(workspace_dir)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Chunk not found: missing" in captured.out


def test_chunks_commands_print_friendly_message_when_chunks_are_missing(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    list_exit_code = main(["chunks", "list", "--workspace", str(workspace_dir)])
    show_exit_code = main(
        ["chunks", "show", "missing", "--workspace", str(workspace_dir)]
    )

    captured = capsys.readouterr()
    assert list_exit_code == 0
    assert show_exit_code == 0
    assert captured.out.count("No chunks found. Run ragent ingest <path> first.") == 2


def test_chunks_command_reports_corrupt_chunks_json(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    chunks_dir = workspace_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("not-json\n", encoding="utf-8")

    exit_code = main(["chunks", "list", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Chunks failed:" in captured.out
    assert "Invalid JSON in chunks file" in captured.out


def test_traces_latest_command_prints_latest_trace_after_ingest(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("abcdefghij", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "5",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(["traces", "latest", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Trace ID: ingest-" in captured.out
    assert "Operation: ingest" in captured.out
    assert "Status: success" in captured.out
    assert "Steps:" in captured.out
    assert "1. load_documents" in captured.out
    assert "2. chunk_documents" in captured.out
    assert "3. write_chunks" in captured.out
    assert "4. write_ingest_summary" in captured.out
    assert "Metadata:" in captured.out
    assert "- source_path:" in captured.out
    assert "- document_count: 1" in captured.out
    assert "- chunk_count: 2" in captured.out


def test_traces_latest_command_prints_friendly_message_when_missing(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["traces", "latest", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No trace found. Run ragent ingest <path> first." in captured.out


def test_traces_list_command_prints_friendly_message_when_missing(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["traces", "list", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No traces found. Run ragent ingest <path> first." in captured.out


def test_traces_list_command_prints_trace_rows_for_operations(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    assert main(["search", "agent", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    assert main(["ask", "agent", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["traces", "list", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Traces" in captured.out
    assert "Trace ID | Operation | Status | Started at | Finished at" in captured.out
    assert "ingest-" in captured.out
    assert "ingest | success" in captured.out
    assert "search-" in captured.out
    assert "search | success" in captured.out
    assert "ask-retrieval-" in captured.out
    assert "ask_retrieval | success" in captured.out


def test_traces_list_command_does_not_print_latest_trace_json_as_trace_id(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["traces", "list", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "latest_trace" not in captured.out


def test_traces_list_command_respects_limit(tmp_path: Path, capsys) -> None:
    workspace_dir = tmp_path / ".ragent"
    traces_dir = workspace_dir / "traces"
    traces_dir.mkdir(parents=True)
    for trace_id, started_at in [
        ("ingest-20260630T115800Z", "2026-06-30T11:58:00Z"),
        ("search-20260630T115900Z", "2026-06-30T11:59:00Z"),
    ]:
        (traces_dir / f"{trace_id}.json").write_text(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "operation": trace_id.split("-", maxsplit=1)[0],
                    "status": "success",
                    "started_at": started_at,
                    "finished_at": started_at,
                    "steps": [],
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        ["traces", "list", "--workspace", str(workspace_dir), "--limit", "1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "search-20260630T115900Z" in captured.out
    assert "ingest-20260630T115800Z" not in captured.out


def test_traces_list_command_prints_warnings_for_corrupt_trace_files(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    traces_dir = workspace_dir / "traces"
    traces_dir.mkdir(parents=True)
    (traces_dir / "search-20260630T115900Z.json").write_text(
        json.dumps(
            {
                "trace_id": "search-20260630T115900Z",
                "operation": "search",
                "status": "success",
                "started_at": "2026-06-30T11:59:00Z",
            }
        ),
        encoding="utf-8",
    )
    (traces_dir / "bad.json").write_text("{not-json", encoding="utf-8")

    exit_code = main(["traces", "list", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "search-20260630T115900Z" in captured.out
    assert "Warnings:" in captured.out
    assert "Skipped invalid trace file:" in captured.out
    assert "bad.json" in captured.out


def test_traces_show_command_prints_detailed_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    trace_files = [
        path
        for path in (workspace_dir / "traces").glob("*.json")
        if path.name != "latest_trace.json"
    ]
    trace_id = json.loads(trace_files[0].read_text(encoding="utf-8"))["trace_id"]

    exit_code = main(
        ["traces", "show", trace_id, "--workspace", str(workspace_dir)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Trace ID: {trace_id}" in captured.out
    assert "Operation: ingest" in captured.out
    assert "Status: success" in captured.out
    assert "Started at:" in captured.out
    assert "Finished at:" in captured.out
    assert "Steps:" in captured.out
    assert "1. load_documents" in captured.out
    assert "Metadata:" in captured.out
    assert "- chunk_count:" in captured.out


def test_traces_show_command_prints_not_found_for_missing_trace(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["traces", "show", "missing", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Trace not found: missing" in captured.out


def test_traces_show_command_reports_corrupt_trace_json(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    traces_dir = workspace_dir / "traces"
    traces_dir.mkdir(parents=True)
    (traces_dir / "bad.json").write_text("{not-json", encoding="utf-8")

    exit_code = main(["traces", "show", "bad", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traces failed:" in captured.out
    assert "Invalid JSON in trace file" in captured.out


def test_traces_latest_command_reports_corrupt_latest_trace_json(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    traces_dir = workspace_dir / "traces"
    traces_dir.mkdir(parents=True)
    (traces_dir / "latest_trace.json").write_text("{not-json", encoding="utf-8")

    exit_code = main(["traces", "latest", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traces failed:" in captured.out
    assert "Invalid JSON in latest trace" in captured.out


def test_index_status_command_prints_missing_index(tmp_path: Path, capsys) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["index", "status", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Semantic index: missing" in captured.out
    assert "Run `ragent index build` to create it." in captured.out


def test_index_build_command_prints_friendly_message_when_chunks_are_missing(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["index", "build", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No chunks found. Run ragent ingest <path> first." in captured.out
    assert not (workspace_dir / "traces" / "latest_trace.json").exists()


def test_index_build_command_requires_embedding_provider(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")

    exit_code = main(["index", "build", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        'Index build failed: embedding provider is not configured. Set [embedding] '
        'provider = "openai_embeddings".'
    ) in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace


def test_index_build_command_writes_vector_index_and_trace_without_api_key(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    calls = install_fake_embedding_post(monkeypatch)

    exit_code = main(["index", "build", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    index_text = (workspace_dir / "index" / "vector_index.jsonl").read_text(
        encoding="utf-8"
    )
    assert exit_code == 0
    assert "Semantic index build" in captured.out
    assert "Embedding provider: openai_embeddings" in captured.out
    assert "Embedding model: text-embedding-3-small" in captured.out
    assert "Chunks embedded: 2" in captured.out
    assert "Embedding dim: 2" in captured.out
    assert "Index path:" in captured.out
    assert "Saved trace to:" in captured.out
    assert (workspace_dir / "index" / "vector_index.jsonl").is_file()
    assert (workspace_dir / "index" / "vector_index_manifest.json").is_file()
    assert "agent memory agent" not in index_text
    assert "embedding-secret-key" not in index_text
    assert "embedding-secret-key" not in captured.out
    assert "embedding-secret-key" not in json.dumps(latest_trace)
    assert latest_trace["operation"] == "index_build"
    assert latest_trace["metadata"]["embedding_provider"] == "openai_embeddings"
    assert latest_trace["metadata"]["embedding_model"] == "text-embedding-3-small"
    assert latest_trace["metadata"]["chunk_count"] == 2
    assert len(calls) == 1


def test_index_status_command_prints_ready_index(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    install_fake_embedding_post(monkeypatch)
    assert main(["index", "build", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["index", "status", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Semantic index: ready" in captured.out
    assert "Index path:" in captured.out
    assert "Chunks indexed: 1" in captured.out
    assert "Embedding model: text-embedding-3-small" in captured.out
    assert "Embedding dim: 2" in captured.out
    assert "Built at:" in captured.out


def test_search_command_prints_matching_chunk_results_after_ingest(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(["search", "agent memory", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Search query: agent memory" in captured.out
    assert "Results: 1" in captured.out
    assert "1. score=3" in captured.out
    assert "rag.md::chunk-0000" in captured.out
    assert "Source:" in captured.out
    assert "Range:" in captured.out
    assert "Preview:" in captured.out
    assert "Saved trace to:" in captured.out
    assert "latest_trace.json" in captured.out

    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert latest_trace["operation"] == "search"
    assert latest_trace["status"] == "success"
    assert latest_trace["metadata"]["query"] == "agent memory"
    assert latest_trace["metadata"]["limit"] == 10
    assert latest_trace["metadata"]["retrieval_mode"] == "lexical"
    assert latest_trace["metadata"]["scoring_method"] == "lexical_token_overlap"
    assert latest_trace["metadata"]["total_chunks"] == 2
    assert latest_trace["metadata"]["result_count"] == 1
    assert len(latest_trace["metadata"]["result_chunk_ids"]) == 1
    assert latest_trace["metadata"]["result_chunk_ids"][0].endswith(
        "rag.md::chunk-0000"
    )


def test_search_command_formats_pdf_page_range_without_char_range(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    write_pdf_chunk_record(workspace_dir)

    exit_code = main(["search", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Range: p.7" in captured.out
    assert "None-None" not in captured.out


def test_search_command_respects_limit(tmp_path: Path, capsys) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics\nagent planning",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "18",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        ["search", "agent", "--workspace", str(workspace_dir), "--limit", "1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Results: 1" in captured.out
    assert "rag.md::chunk-0000" in captured.out
    assert "rag.md::chunk-0002" not in captured.out


def test_search_command_explicit_lexical_mode_works(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "search",
            "agent",
            "--retrieval",
            "lexical",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Search query: agent" in captured.out
    assert "Retrieval mode: lexical" in captured.out
    assert "rag.md::chunk-0000" in captured.out


def test_search_command_semantic_mode_works_after_index_build(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    install_fake_embedding_post(monkeypatch)
    assert main(["index", "build", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "search",
            "retrieval",
            "--retrieval",
            "semantic",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Search query: retrieval" in captured.out
    assert "Retrieval mode: semantic" in captured.out
    assert "Results: 2" in captured.out
    assert "1. score=" in captured.out
    assert "rag.md::chunk-0001" in captured.out
    assert "embedding-secret-key" not in captured.out
    assert latest_trace["operation"] == "search"
    assert latest_trace["metadata"]["retrieval_mode"] == "semantic"
    assert latest_trace["metadata"]["retrieval_method"] == "semantic_cosine_similarity"
    assert latest_trace["metadata"]["embedding_provider"] == "openai_embeddings"
    assert latest_trace["metadata"]["embedding_model"] == "text-embedding-3-small"
    assert [step["name"] for step in latest_trace["steps"]] == [
        "read_chunks",
        "embed_query",
        "load_vector_index",
        "score_vectors",
        "rank_results",
        "render_results",
    ]
    assert "embedding-secret-key" not in json.dumps(latest_trace)


def test_search_command_semantic_mode_missing_index_fails_without_new_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "search",
            "agent",
            "--retrieval",
            "semantic",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        "Semantic search failed: vector index not found. "
        "Run `ragent index build` first."
    ) in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace


def test_search_command_hybrid_mode_works_after_index_build(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    install_fake_embedding_post(monkeypatch)
    assert main(["index", "build", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "search",
            "retrieval",
            "--retrieval",
            "hybrid",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Search query: retrieval" in captured.out
    assert "Retrieval mode: hybrid" in captured.out
    assert "Results:" in captured.out
    assert "Saved trace to:" in captured.out
    assert "rag.md::chunk-0001" in captured.out
    assert "embedding-secret-key" not in captured.out
    assert latest_trace["operation"] == "search"
    assert latest_trace["metadata"]["retrieval_mode"] == "hybrid"
    assert latest_trace["metadata"]["retrieval_method"] == "hybrid_rrf"
    assert latest_trace["metadata"]["fusion_method"] == "reciprocal_rank_fusion"
    assert latest_trace["metadata"]["rrf_k"] == 60
    assert latest_trace["metadata"]["lexical_weight"] == 1.0
    assert latest_trace["metadata"]["semantic_weight"] == 1.0
    assert latest_trace["metadata"]["candidate_limit"] == 40
    assert latest_trace["metadata"]["embedding_provider"] == "openai_embeddings"
    assert latest_trace["metadata"]["embedding_model"] == "text-embedding-3-small"
    assert latest_trace["metadata"]["index_path"].endswith("vector_index.jsonl")
    assert [step["name"] for step in latest_trace["steps"]] == [
        "read_chunks",
        "run_lexical_search",
        "embed_query",
        "load_vector_index",
        "run_semantic_search",
        "fuse_results",
        "rank_results",
        "render_results",
    ]
    assert "embedding-secret-key" not in json.dumps(latest_trace)
    assert "retrieval basics" not in json.dumps(latest_trace)


def test_search_command_hybrid_mode_missing_index_fails_without_new_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "search",
            "agent",
            "--retrieval",
            "hybrid",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        "Hybrid search failed: vector index not found. "
        "Run `ragent index build` first."
    ) in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace


def test_eval_generate_parser_accepts_expected_args() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "generate",
            "--source",
            "knowledge",
            "--workspace",
            ".ragent",
            "--output",
            "cases.jsonl",
            "--questions-per-span",
            "3",
            "--max-cases",
            "9",
            "--min-evidence-chars",
            "20",
            "--max-evidence-chars",
            "900",
            "--include-pdf",
            "--overwrite",
            "--dry-run",
        ]
    )

    assert args.command == "eval"
    assert args.eval_command == "generate"
    assert args.source == "knowledge"
    assert args.workspace == ".ragent"
    assert args.output == "cases.jsonl"
    assert args.questions_per_span == 3
    assert args.max_cases == 9
    assert args.min_evidence_chars == 20
    assert args.max_evidence_chars == 900
    assert args.include_pdf is True
    assert args.overwrite is True
    assert args.dry_run is True


def test_eval_generate_dry_run_extracts_spans_without_calling_generator(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "# Guide\n\n"
        "Hybrid retrieval keeps evidence spans independent from chunk ids.",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    output_path = tmp_path / "cases.jsonl"

    def fail_build_text_generation_client(config):
        raise AssertionError("dry-run must not build a generator")

    monkeypatch.setattr(
        "ragent_forge.cli._build_text_generation_client",
        fail_build_text_generation_client,
    )

    exit_code = main(
        [
            "eval",
            "generate",
            "--source",
            str(knowledge_dir),
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_path),
            "--questions-per-span",
            "2",
            "--max-cases",
            "5",
            "--min-evidence-chars",
            "20",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Eval dataset generation dry run" in captured.out
    assert "Evidence spans extracted: 1" in captured.out
    assert "include_pdf: False" in captured.out
    assert "questions_per_span: 2" in captured.out
    assert "max_cases: 5" in captured.out
    assert "Estimated max generated cases: 2" in captured.out
    assert not output_path.exists()


def test_eval_generate_null_provider_fails_before_generation(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "# Guide\n\n"
        "Hybrid retrieval keeps evidence spans independent from chunk ids.",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    output_path = tmp_path / "cases.jsonl"

    def fail_build_text_generation_client(config):
        raise AssertionError("null provider must not build a generator")

    monkeypatch.setattr(
        "ragent_forge.cli._build_text_generation_client",
        fail_build_text_generation_client,
    )

    exit_code = main(
        [
            "eval",
            "generate",
            "--source",
            str(knowledge_dir),
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_path),
            "--min-evidence-chars",
            "20",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Eval generation failed:" in captured.out
    assert "generation provider is not configured" in captured.out
    assert not output_path.exists()


def test_eval_generate_writes_span_based_jsonl(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "# Guide\n\n"
        "Hybrid retrieval combines lexical and semantic retrieval while "
        "keeping source evidence inspectable.",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    write_generation_config(workspace_dir)
    output_path = tmp_path / "generated_cases.jsonl"
    fake_generator = FakeEvalTextGenerationClient(
        [
            json.dumps(
                {
                    "items": [
                        {
                            "query": "How does hybrid retrieval use evidence?",
                            "reference_answer": (
                                "It combines lexical and semantic retrieval "
                                "while keeping source evidence inspectable."
                            ),
                            "question_type": "reasoning",
                            "difficulty": "medium",
                        }
                    ]
                }
            )
        ]
    )

    def build_fake_text_generation_client(config):
        assert config.generation.provider == "openai_responses"
        return fake_generator

    monkeypatch.setattr(
        "ragent_forge.cli._build_text_generation_client",
        build_fake_text_generation_client,
    )

    exit_code = main(
        [
            "eval",
            "generate",
            "--source",
            str(knowledge_dir),
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_path),
            "--questions-per-span",
            "1",
            "--min-evidence-chars",
            "20",
        ]
    )

    captured = capsys.readouterr()
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    loaded_cases = RetrievalEvalService().load_cases(output_path)
    assert exit_code == 0
    assert "Eval dataset generation" in captured.out
    assert "Cases generated: 1" in captured.out
    assert "ragent eval retrieval --cases" in captured.out
    assert len(fake_generator.calls) == 1
    assert fake_generator.calls[0][1] == (
        "You are generating retrieval evaluation cases for a RAG system."
    )
    assert len(records) == 1
    assert records[0]["id"] == "synthetic-span-000001"
    assert "evidence_spans" in records[0]
    assert "expected_chunk_ids" not in records[0]
    assert "expected_source_paths" not in records[0]
    assert loaded_cases[0].evidence_spans[0].source_path.endswith("rag.md")
    assert loaded_cases[0].expected_chunk_ids == []


def test_eval_generate_fails_when_no_cases_are_generated(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "# Guide\n\n"
        "Hybrid retrieval combines lexical and semantic retrieval while "
        "keeping source evidence inspectable.",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    write_generation_config(workspace_dir)
    output_path = tmp_path / "generated_cases.jsonl"
    fake_generator = FakeEvalTextGenerationClient(["not-json"])

    def build_fake_text_generation_client(config):
        assert config.generation.provider == "openai_responses"
        return fake_generator

    monkeypatch.setattr(
        "ragent_forge.cli._build_text_generation_client",
        build_fake_text_generation_client,
    )

    exit_code = main(
        [
            "eval",
            "generate",
            "--source",
            str(knowledge_dir),
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_path),
            "--questions-per-span",
            "1",
            "--min-evidence-chars",
            "20",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert not output_path.exists()
    assert "Eval generation failed: no eval cases were generated." in captured.out
    assert "Spans skipped: 1" in captured.out
    assert "Error count: 1" in captured.out
    assert "invalid JSON response" in captured.out


def test_eval_generate_respects_overwrite_flag(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "# Guide\n\nEvidence spans should be written to JSONL.",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    write_generation_config(workspace_dir)
    output_path = tmp_path / "generated_cases.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")
    fake_generators: list[FakeEvalTextGenerationClient] = []

    def build_fake_text_generation_client(config):
        fake_generator = FakeEvalTextGenerationClient(
            [
                json.dumps(
                    {
                        "items": [
                            {
                                "query": "What gets written?",
                                "reference_answer": "Evidence spans are written.",
                                "question_type": "factual",
                                "difficulty": "easy",
                            }
                        ]
                    }
                )
            ]
        )
        fake_generators.append(fake_generator)
        return fake_generator

    monkeypatch.setattr(
        "ragent_forge.cli._build_text_generation_client",
        build_fake_text_generation_client,
    )

    blocked_exit_code = main(
        [
            "eval",
            "generate",
            "--source",
            str(knowledge_dir),
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_path),
            "--questions-per-span",
            "1",
            "--min-evidence-chars",
            "20",
        ]
    )
    blocked_output = capsys.readouterr()
    blocked_output_text = output_path.read_text(encoding="utf-8")
    blocked_fake_generators = list(fake_generators)

    overwrite_exit_code = main(
        [
            "eval",
            "generate",
            "--source",
            str(knowledge_dir),
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_path),
            "--questions-per-span",
            "1",
            "--min-evidence-chars",
            "20",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    assert blocked_exit_code == 1
    assert "Output JSONL already exists" in blocked_output.out
    assert blocked_output_text == "existing\n"
    assert blocked_fake_generators == []
    assert overwrite_exit_code == 0
    assert "Cases generated: 1" in captured.out
    assert json.loads(output_path.read_text(encoding="utf-8"))["id"] == (
        "synthetic-span-000001"
    )
    assert len(fake_generators) == 1


def test_eval_generate_pdf_is_skipped_by_default_and_attempted_when_enabled(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    markdown = knowledge_dir / "guide.md"
    markdown.write_text(
        "# Guide\n\nMarkdown evidence should be extracted for eval generation.",
        encoding="utf-8",
    )
    pdf = knowledge_dir / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nnot a real pdf")
    workspace_dir = tmp_path / ".ragent"
    output_path = tmp_path / "generated_cases.jsonl"
    real_load_structured_document = evidence_span_service.load_structured_document
    loaded_file_names: list[str] = []

    def recording_load_structured_document(path: str | Path):
        path = Path(path)
        loaded_file_names.append(path.name)
        if path.suffix.lower() == ".pdf":
            raise RuntimeError("PDF loading attempted")
        return real_load_structured_document(path)

    monkeypatch.setattr(
        evidence_span_service,
        "load_structured_document",
        recording_load_structured_document,
    )

    default_exit_code = main(
        [
            "eval",
            "generate",
            "--source",
            str(knowledge_dir),
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_path),
            "--min-evidence-chars",
            "20",
            "--dry-run",
        ]
    )
    default_output = capsys.readouterr()
    default_loaded_file_names = list(loaded_file_names)

    loaded_file_names.clear()
    include_pdf_exit_code = main(
        [
            "eval",
            "generate",
            "--source",
            str(knowledge_dir),
            "--workspace",
            str(workspace_dir),
            "--output",
            str(output_path),
            "--min-evidence-chars",
            "20",
            "--include-pdf",
            "--dry-run",
        ]
    )

    include_pdf_output = capsys.readouterr()
    assert default_exit_code == 0
    assert "Evidence spans extracted: 1" in default_output.out
    assert default_loaded_file_names == ["guide.md"]
    assert loaded_file_names == ["guide.md", "paper.pdf"]
    assert include_pdf_exit_code == 1
    assert "PDF loading attempted" in include_pdf_output.out


def test_eval_retrieval_defaults_to_lexical_and_writes_report_and_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    chunks = [
        json.loads(line)
        for line in (workspace_dir / "chunks" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    write_retrieval_eval_cases(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "agent memory",
                "expected_chunk_ids": [chunks[0]["chunk_id"]],
            },
            {
                "id": "case-002",
                "query": "missing",
                "expected_source_paths": ["missing.md"],
            },
        ],
    )

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    report_path = workspace_dir / "eval" / "latest_retrieval_eval.json"
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Retrieval evaluation" in captured.out
    assert "Cases: 2" in captured.out
    assert "Retrieval mode: lexical" in captured.out
    assert "Limit: 5" in captured.out
    assert "Passed: 1" in captured.out
    assert "Failed: 1" in captured.out
    assert "hit@1: 0.5000" in captured.out
    assert "hit@3: 0.5000" in captured.out
    assert "hit@5 requested: 0.5000" in captured.out
    assert "MRR: 0.5000" in captured.out
    assert "recall@5 requested:" in captured.out
    assert "Avg retrieval latency:" in captured.out
    assert "Failed cases:" in captured.out
    assert "- case-002 | rank: none | query: missing" in captured.out
    assert "Failure breakdown:" in captured.out
    assert "- no_result: 1" in captured.out
    assert "Report path:" in captured.out
    assert "Run directory:" in captured.out
    assert "Saved trace to:" in captured.out
    assert report_path.is_file()
    run_dirs = list((workspace_dir / "eval" / "runs").glob("retrieval-*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "summary.md").is_file()
    assert (run_dir / "cases.jsonl").is_file()
    assert (run_dir / "failures.jsonl").is_file()
    run_summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    run_cases = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    run_failures = [
        json.loads(line)
        for line in (run_dir / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert report["evaluation_type"] == "retrieval"
    assert report["retrieval_mode"] == "lexical"
    assert report["retrieval_method"] == "lexical_token_overlap"
    assert report["case_count"] == 2
    assert report["passed_count"] == 1
    assert report["failed_count"] == 1
    assert report["metrics"]["hit@k"] == 0.5
    assert "recall@k" in report["metrics"]
    assert "avg_retrieval_latency_ms" in report["metrics"]
    assert report["results"][0]["failure_type"] is None
    assert report["results"][0]["failure_reason"] is None
    assert report["results"][1]["failure_type"] == "no_result"
    assert (
        report["results"][1]["failure_reason"]
        == "No retrieval results returned."
    )
    assert run_summary == report
    assert [record["id"] for record in run_cases] == ["case-001", "case-002"]
    assert [record["id"] for record in run_failures] == ["case-002"]
    assert set(run_cases[0]) == {
        "id",
        "query",
        "passed",
        "failure_type",
        "failure_reason",
        "rank",
        "matched_by",
        "expected_chunk_ids",
        "expected_source_paths",
        "actual_chunk_ids",
        "actual_source_paths",
        "metadata",
    }
    assert latest_trace["operation"] == "retrieval_eval"
    assert latest_trace["metadata"]["retrieval_mode"] == "lexical"
    assert latest_trace["metadata"]["case_count"] == 2
    assert "embedding_provider" not in latest_trace["metadata"]
    assert "agent memory agent" not in report_text
    assert "agent memory agent" not in (run_dir / "cases.jsonl").read_text(
        encoding="utf-8"
    )
    assert "agent memory agent" not in (run_dir / "failures.jsonl").read_text(
        encoding="utf-8"
    )
    assert "agent memory agent" not in json.dumps(latest_trace)


def test_eval_retrieval_semantic_succeeds_after_index_build(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    install_fake_embedding_post(monkeypatch)
    assert main(["index", "build", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    chunks = [
        json.loads(line)
        for line in (workspace_dir / "chunks" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    write_retrieval_eval_cases(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "retrieval",
                "expected_chunk_ids": [chunks[1]["chunk_id"]],
            }
        ],
    )

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--retrieval",
            "semantic",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    report_text = (workspace_dir / "eval" / "latest_retrieval_eval.json").read_text(
        encoding="utf-8"
    )
    report = json.loads(report_text)
    assert exit_code == 0
    assert "Retrieval mode: semantic" in captured.out
    assert "Failed cases: none" in captured.out
    assert "embedding-secret-key" not in captured.out
    assert report["retrieval_method"] == "semantic_cosine_similarity"
    assert report["embedding_provider"] == "openai_embeddings"
    assert report["embedding_model"] == "text-embedding-3-small"
    assert report["index_path"].endswith("vector_index.jsonl")
    assert latest_trace["operation"] == "retrieval_eval"
    assert latest_trace["metadata"]["retrieval_mode"] == "semantic"
    assert latest_trace["metadata"]["embedding_provider"] == "openai_embeddings"
    assert latest_trace["metadata"]["embedding_model"] == "text-embedding-3-small"
    assert latest_trace["metadata"]["index_path"].endswith("vector_index.jsonl")
    assert "embedding-secret-key" not in report_text
    assert '"embedding": [' not in report_text
    assert "retrieval basics" not in report_text
    assert "embedding-secret-key" not in json.dumps(latest_trace)
    assert '"embedding": [' not in json.dumps(latest_trace)


def test_eval_retrieval_semantic_missing_index_fails_without_new_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    write_retrieval_eval_cases(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "agent",
                "expected_source_paths": ["rag.md"],
            }
        ],
    )
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--retrieval",
            "semantic",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        "Retrieval eval failed: vector index not found. "
        "Run `ragent index build` first."
    ) in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace
    assert not (workspace_dir / "eval" / "latest_retrieval_eval.json").exists()


def test_eval_retrieval_hybrid_succeeds_after_index_build(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    install_fake_embedding_post(monkeypatch)
    assert main(["index", "build", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    chunks = [
        json.loads(line)
        for line in (workspace_dir / "chunks" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    write_retrieval_eval_cases(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "retrieval",
                "expected_chunk_ids": [chunks[1]["chunk_id"]],
            }
        ],
    )

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--retrieval",
            "hybrid",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    report_text = (workspace_dir / "eval" / "latest_retrieval_eval.json").read_text(
        encoding="utf-8"
    )
    report = json.loads(report_text)
    assert exit_code == 0
    assert "Retrieval mode: hybrid" in captured.out
    assert "Failed cases: none" in captured.out
    assert "embedding-secret-key" not in captured.out
    assert report["retrieval_mode"] == "hybrid"
    assert report["retrieval_method"] == "hybrid_rrf"
    assert report["fusion_method"] == "reciprocal_rank_fusion"
    assert report["rrf_k"] == 60
    assert report["lexical_weight"] == 1.0
    assert report["semantic_weight"] == 1.0
    assert report["embedding_provider"] == "openai_embeddings"
    assert report["embedding_model"] == "text-embedding-3-small"
    assert report["index_path"].endswith("vector_index.jsonl")
    assert report["metrics"]["hit@k"] == 1.0
    assert latest_trace["operation"] == "retrieval_eval"
    assert latest_trace["metadata"]["retrieval_mode"] == "hybrid"
    assert latest_trace["metadata"]["retrieval_method"] == "hybrid_rrf"
    assert latest_trace["metadata"]["fusion_method"] == "reciprocal_rank_fusion"
    assert latest_trace["metadata"]["rrf_k"] == 60
    assert latest_trace["metadata"]["embedding_provider"] == "openai_embeddings"
    assert latest_trace["metadata"]["embedding_model"] == "text-embedding-3-small"
    assert latest_trace["metadata"]["index_path"].endswith("vector_index.jsonl")
    assert "embedding-secret-key" not in report_text
    assert '"embedding": [' not in report_text
    assert "retrieval basics" not in report_text
    assert "embedding-secret-key" not in json.dumps(latest_trace)
    assert '"embedding": [' not in json.dumps(latest_trace)


def test_eval_retrieval_hybrid_missing_index_fails_without_new_trace_or_report(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    write_retrieval_eval_cases(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "agent",
                "expected_source_paths": ["rag.md"],
            }
        ],
    )
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--retrieval",
            "hybrid",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        "Retrieval eval failed: vector index not found. "
        "Run `ragent index build` first."
    ) in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace
    assert not (workspace_dir / "eval" / "latest_retrieval_eval.json").exists()


def test_eval_retrieval_missing_cases_file_fails_clearly(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(tmp_path / "missing.jsonl"),
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Retrieval eval failed: cases file not found:" in captured.out
    assert not (workspace_dir / "traces" / "latest_trace.json").exists()


def test_eval_retrieval_missing_chunks_fails_without_trace(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    write_retrieval_eval_cases(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "agent",
                "expected_source_paths": ["rag.md"],
            }
        ],
    )

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        "Retrieval eval failed: no chunks found. "
        "Run ragent ingest <path> first."
    ) in captured.out
    assert not (workspace_dir / "traces" / "latest_trace.json").exists()


def test_eval_retrieval_invalid_cases_jsonl_fails_clearly_without_new_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    cases_path.parent.mkdir(parents=True)
    cases_path.write_text("{not-json}\n", encoding="utf-8")

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Retrieval eval failed:" in captured.out
    assert "line 1" in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace


def test_eval_retrieval_empty_cases_file_fails_clearly(tmp_path: Path, capsys) -> None:
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    cases_path.parent.mkdir(parents=True)
    cases_path.write_text("\n", encoding="utf-8")

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--workspace",
            str(tmp_path / ".ragent"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Retrieval eval failed: no eval cases found" in captured.out


def test_eval_retrieval_limit_zero_fails_clearly(tmp_path: Path, capsys) -> None:
    cases_path = tmp_path / "eval" / "retrieval_cases.jsonl"
    write_retrieval_eval_cases(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "agent",
                "expected_source_paths": ["rag.md"],
            }
        ],
    )

    exit_code = main(
        [
            "eval",
            "retrieval",
            "--cases",
            str(cases_path),
            "--limit",
            "0",
            "--workspace",
            str(tmp_path / ".ragent"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Retrieval eval failed: limit must be greater than 0" in captured.out


def test_search_command_prints_no_matches(tmp_path: Path, capsys) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["search", "no-match", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Search query: no-match" in captured.out
    assert "No matches found." in captured.out
    assert "Saved trace to:" in captured.out

    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert latest_trace["operation"] == "search"
    assert latest_trace["status"] == "success"
    assert latest_trace["metadata"]["query"] == "no-match"
    assert latest_trace["metadata"]["retrieval_mode"] == "lexical"
    assert latest_trace["metadata"]["result_count"] == 0
    assert latest_trace["metadata"]["result_chunk_ids"] == []


def test_search_command_prints_friendly_message_when_chunks_are_missing(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["search", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No chunks found. Run ragent ingest <path> first." in captured.out
    assert not (workspace_dir / "traces" / "latest_trace.json").exists()


def test_search_command_reports_corrupt_chunks_json(tmp_path: Path, capsys) -> None:
    workspace_dir = tmp_path / ".ragent"
    chunks_dir = workspace_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("not-json\n", encoding="utf-8")

    exit_code = main(["search", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Search failed:" in captured.out
    assert "Invalid JSON in chunks file" in captured.out
    assert not (workspace_dir / "traces" / "latest_trace.json").exists()


def test_traces_latest_command_prints_latest_trace_after_search(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    assert main(["search", "agent", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["traces", "latest", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Trace ID: search-" in captured.out
    assert "Operation: search" in captured.out
    assert "Status: success" in captured.out
    assert "1. read_chunks" in captured.out
    assert "2. tokenize_query" in captured.out
    assert "- query: agent" in captured.out


def test_ask_command_prints_retrieval_only_context_after_ingest(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(["ask", "agent memory", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Ask pipeline: retrieval-only mode" in captured.out
    assert "Question: agent memory" in captured.out
    assert "Generation: not configured." in captured.out
    assert "Answer:" not in captured.out
    assert "Generated answer:" not in captured.out
    assert "Retrieved context:" in captured.out
    assert "1. score=3" in captured.out
    assert "rag.md::chunk-0000" in captured.out
    assert "Source:" in captured.out
    assert "Range:" in captured.out
    assert "Preview:" in captured.out
    assert "Saved trace to:" in captured.out

    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert latest_trace["operation"] == "ask_retrieval"
    assert latest_trace["status"] == "success"
    assert latest_trace["metadata"]["question"] == "agent memory"
    assert latest_trace["metadata"]["limit"] == 5
    assert latest_trace["metadata"]["retrieval_mode"] == "lexical"
    assert latest_trace["metadata"]["retrieval_method"] == "lexical_token_overlap"
    assert latest_trace["metadata"]["total_chunks"] == 2
    assert latest_trace["metadata"]["retrieved_count"] == 1
    assert len(latest_trace["metadata"]["retrieved_chunk_ids"]) == 1
    assert latest_trace["metadata"]["retrieved_chunk_ids"][0].endswith(
        "rag.md::chunk-0000"
    )
    assert latest_trace["metadata"]["generation_status"] == "not_implemented"
    assert latest_trace["metadata"]["generation_provider"] == "null"
    assert latest_trace["metadata"]["generation_result_status"] == "not_configured"
    assert latest_trace["metadata"]["answer_generated"] is False
    assert latest_trace["metadata"]["config_generation_provider"] == "null"


def test_ask_command_semantic_mode_uses_semantic_results_after_index_build(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    install_fake_embedding_post(monkeypatch)
    assert main(["index", "build", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "ask",
            "retrieval",
            "--retrieval",
            "semantic",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Ask pipeline: retrieval-only mode" in captured.out
    assert "Question: retrieval" in captured.out
    assert "Retrieval mode: semantic" in captured.out
    assert "rag.md::chunk-0001" in captured.out
    assert "Generation: not configured." in captured.out
    assert "embedding-secret-key" not in captured.out
    assert latest_trace["operation"] == "ask_retrieval"
    assert latest_trace["metadata"]["retrieval_mode"] == "semantic"
    assert latest_trace["metadata"]["retrieval_method"] == "semantic_cosine_similarity"
    assert latest_trace["metadata"]["embedding_provider"] == "openai_embeddings"
    assert latest_trace["metadata"]["embedding_model"] == "text-embedding-3-small"
    assert latest_trace["metadata"]["retrieved_chunk_ids"][0].endswith(
        "rag.md::chunk-0001"
    )
    retrieve_step = latest_trace["steps"][1]
    assert retrieve_step["name"] == "retrieve_context"
    assert "semantic vector search" in retrieve_step["description"]
    assert "lexical search" not in retrieve_step["description"]
    assert "embedding-secret-key" not in json.dumps(latest_trace)


def test_ask_command_semantic_mode_missing_index_fails_without_new_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "ask",
            "agent",
            "--retrieval",
            "semantic",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        "Ask failed: vector index not found. Run `ragent index build` first."
    ) in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace


def test_ask_command_hybrid_mode_uses_hybrid_results_after_index_build(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    install_fake_embedding_post(monkeypatch)
    assert main(["index", "build", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "ask",
            "retrieval",
            "--retrieval",
            "hybrid",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Ask pipeline: retrieval-only mode" in captured.out
    assert "Question: retrieval" in captured.out
    assert "Retrieval mode: hybrid" in captured.out
    assert "rag.md::chunk-0001" in captured.out
    assert "Generation: not configured." in captured.out
    assert "embedding-secret-key" not in captured.out
    assert latest_trace["operation"] == "ask_retrieval"
    assert latest_trace["metadata"]["retrieval_mode"] == "hybrid"
    assert latest_trace["metadata"]["retrieval_method"] == "hybrid_rrf"
    assert latest_trace["metadata"]["fusion_method"] == "reciprocal_rank_fusion"
    assert latest_trace["metadata"]["rrf_k"] == 60
    assert latest_trace["metadata"]["lexical_weight"] == 1.0
    assert latest_trace["metadata"]["semantic_weight"] == 1.0
    assert latest_trace["metadata"]["embedding_provider"] == "openai_embeddings"
    assert latest_trace["metadata"]["embedding_model"] == "text-embedding-3-small"
    assert latest_trace["metadata"]["index_path"].endswith("vector_index.jsonl")
    assert latest_trace["metadata"]["retrieved_chunk_ids"][0].endswith(
        "rag.md::chunk-0001"
    )
    retrieve_step = latest_trace["steps"][1]
    assert retrieve_step["name"] == "retrieve_context"
    assert retrieve_step["description"] == (
        "Retrieve context chunks with hybrid lexical and semantic search."
    )
    assert retrieve_step["inputs"]["retrieval_method"] == "hybrid_rrf"
    assert retrieve_step["inputs"]["fusion_method"] == "reciprocal_rank_fusion"
    assert retrieve_step["inputs"]["rrf_k"] == 60
    assert "embedding-secret-key" not in json.dumps(latest_trace)
    assert "retrieval basics" not in json.dumps(latest_trace)


def test_ask_command_hybrid_mode_missing_index_fails_without_new_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    write_embedding_config(workspace_dir)
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "ask",
            "agent",
            "--retrieval",
            "hybrid",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        "Ask failed: vector index not found. Run `ragent index build` first."
    ) in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace


def test_ask_command_show_prompt_prints_context_pack_and_prompt_preview(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        ["ask", "agent memory", "--show-prompt", "--workspace", str(workspace_dir)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Ask pipeline: retrieval-only mode" in captured.out
    assert "Retrieved context:" in captured.out
    assert "Context pack:" in captured.out
    assert "Context chunks: 1" in captured.out
    assert "Total context chars:" in captured.out
    assert "Retrieval method: lexical_token_overlap" in captured.out
    assert "Generation prompt:" in captured.out
    assert (
        "You are RAGentForge, a local retrieval-augmented assistant."
        in captured.out
    )
    assert "Question:\nagent memory" in captured.out
    assert (
        "Use only the retrieved context below to answer the question."
        in captured.out
    )
    assert "[1] Source:" in captured.out
    assert "Chunk ID:" in captured.out
    assert "Content:" in captured.out
    assert "agent memory agent" in captured.out
    assert "Saved trace to:" in captured.out

    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert latest_trace["operation"] == "ask_retrieval"
    assert latest_trace["metadata"]["context_chunk_count"] == 1
    assert latest_trace["metadata"]["total_context_chars"] > 0
    assert latest_trace["metadata"]["prompt_preview_shown"] is True
    assert latest_trace["metadata"]["max_context_chars"] == 4000
    assert latest_trace["metadata"]["generation_provider"] == "null"
    assert latest_trace["metadata"]["generation_result_status"] == "not_configured"
    assert latest_trace["metadata"]["answer_generated"] is False
    assert latest_trace["metadata"]["config_generation_provider"] == "null"


def test_ask_command_show_prompt_respects_max_context_chars(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent abcdefghijklmnopqrstuvwxyz",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "ask",
            "agent",
            "--show-prompt",
            "--max-context-chars",
            "20",
            "--workspace",
            str(workspace_dir),
        ]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Total context chars: 20" in captured.out
    prompt_preview = captured.out.split("Generation prompt:", maxsplit=1)[1]
    prompt_preview = prompt_preview.split("Saved trace to:", maxsplit=1)[0]
    assert "Content:\nagent abcdefghijklmn" in prompt_preview
    assert "Content:\nagent abcdefghijklmno" not in prompt_preview
    assert latest_trace["metadata"]["total_context_chars"] == 20
    assert latest_trace["metadata"]["max_context_chars"] == 20


def test_ask_command_without_show_prompt_stays_compact(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["ask", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Retrieved context:" in captured.out
    assert "Context pack:" not in captured.out
    assert "Generation prompt:" not in captured.out
    assert (
        "You are RAGentForge, a local retrieval-augmented assistant."
        not in captured.out
    )
    assert latest_trace["metadata"]["prompt_preview_shown"] is False
    assert latest_trace["metadata"]["max_context_chars"] == 4000


def test_ask_command_no_match_show_prompt_prints_empty_context_pack(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        ["ask", "no-match", "--show-prompt", "--workspace", str(workspace_dir)]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "No retrieved context found." in captured.out
    assert "Context pack:" in captured.out
    assert "Context chunks: 0" in captured.out
    assert "Total context chars: 0" in captured.out
    assert "Generation prompt:" in captured.out
    assert "Retrieved context:\nNo retrieved context." in captured.out
    assert latest_trace["metadata"]["retrieved_count"] == 0
    assert latest_trace["metadata"]["context_chunk_count"] == 0
    assert latest_trace["metadata"]["total_context_chars"] == 0
    assert latest_trace["metadata"]["prompt_preview_shown"] is True


def test_ask_command_respects_limit(tmp_path: Path, capsys) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics\nagent planning",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "18",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        ["ask", "agent", "--workspace", str(workspace_dir), "--limit", "1"]
    )

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Retrieved context:" in captured.out
    assert "rag.md::chunk-0000" in captured.out
    assert "rag.md::chunk-0002" not in captured.out
    assert latest_trace["metadata"]["limit"] == 1
    assert latest_trace["metadata"]["retrieved_count"] == 1


def test_ask_command_prints_no_retrieved_context_and_writes_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["ask", "no-match", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Ask pipeline: retrieval-only mode" in captured.out
    assert "Question: no-match" in captured.out
    assert "Generation skipped because there is no retrieved context." in captured.out
    assert "Answer:" not in captured.out
    assert "Generated answer:" not in captured.out
    assert "No retrieved context found." in captured.out
    assert "Saved trace to:" in captured.out

    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert latest_trace["operation"] == "ask_retrieval"
    assert latest_trace["metadata"]["question"] == "no-match"
    assert latest_trace["metadata"]["retrieved_count"] == 0
    assert latest_trace["metadata"]["retrieved_chunk_ids"] == []
    assert latest_trace["metadata"]["generation_status"] == "not_implemented"
    assert latest_trace["metadata"]["generation_provider"] == "null"
    assert latest_trace["metadata"]["generation_result_status"] == "skipped"
    assert latest_trace["metadata"]["answer_generated"] is False
    assert latest_trace["metadata"]["config_generation_provider"] == "null"
    assert latest_trace["metadata"]["skip_reason"] == "no_retrieved_context"


def test_ask_command_works_with_explicit_null_config(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    (workspace_dir / "config.toml").write_text(
        '[generation]\nprovider = "null"\n',
        encoding="utf-8",
    )

    exit_code = main(["ask", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Ask pipeline: retrieval-only mode" in captured.out
    assert latest_trace["operation"] == "ask_retrieval"
    assert latest_trace["metadata"]["generation_provider"] == "null"
    assert latest_trace["metadata"]["config_generation_provider"] == "null"


def test_ask_command_rejects_unsupported_generation_provider_without_ask_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")
    (workspace_dir / "config.toml").write_text(
        '[generation]\nprovider = "openai"\n',
        encoding="utf-8",
    )

    exit_code = main(["ask", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    latest_trace = json.loads(latest_trace_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert "Ask failed:" in captured.out
    assert "Unsupported generation provider: openai" in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace
    assert latest_trace["operation"] == "ingest"


def test_ask_command_with_openai_responses_prints_answer_sources_and_trace_metadata(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text(
        "agent memory agent\nretrieval basics",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / ".ragent"
    assert (
        main(
            [
                "ingest",
                str(knowledge_dir),
                "--chunk-size",
                "20",
                "--workspace",
                str(workspace_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    (workspace_dir / "config.toml").write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = "super-secret-key"\n'
            "timeout_seconds = 60\n"
            "temperature = 0.2\n"
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"output_text": "Generated answer from provider"}

    call_count = 0

    def fake_post(url, *, headers, json, timeout):
        nonlocal call_count
        call_count += 1
        assert url == "https://api.openai.com/v1/responses"
        assert headers["Authorization"] == "Bearer super-secret-key"
        assert json["model"] == "gpt-4o-mini"
        assert json["temperature"] == 0.2
        return FakeResponse()

    monkeypatch.setattr(
        "ragent_forge.app.services.generation_service.httpx.post",
        fake_post,
    )

    exit_code = main(["ask", "agent memory", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Ask pipeline: generated answer mode" in captured.out
    assert "Generation provider: openai_responses" in captured.out
    assert "Generation status: success" in captured.out
    assert "Answer:" in captured.out
    assert "Generated answer from provider" in captured.out
    assert "Sources:" in captured.out
    assert "rag.md::chunk-0000" in captured.out
    assert "Score: 3" in captured.out
    assert "super-secret-key" not in captured.out
    assert "Retrieved context:" not in captured.out
    assert call_count == 1
    assert latest_trace["metadata"]["generation_provider"] == "openai_responses"
    assert latest_trace["metadata"]["generation_status"] == "generated"
    assert latest_trace["metadata"]["generation_result_status"] == "success"
    assert latest_trace["metadata"]["answer_generated"] is True
    assert latest_trace["metadata"]["model"] == "gpt-4o-mini"
    assert latest_trace["metadata"]["base_url"] == "https://api.openai.com/v1"
    assert latest_trace["metadata"]["endpoint"] == "/responses"
    assert latest_trace["metadata"]["source_count"] == 1
    assert "super-secret-key" not in json.dumps(latest_trace)


def test_ask_command_openai_no_match_skips_provider_and_keeps_trace_safe(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    (workspace_dir / "config.toml").write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = "super-secret-key"\n'
        ),
        encoding="utf-8",
    )

    def fake_post(url, *, headers, json, timeout):
        raise AssertionError("provider should not be called")

    monkeypatch.setattr(
        "ragent_forge.app.services.generation_service.httpx.post",
        fake_post,
    )

    exit_code = main(["ask", "no-match", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    latest_trace = json.loads(
        (workspace_dir / "traces" / "latest_trace.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert "Generation skipped because there is no retrieved context." in captured.out
    assert latest_trace["metadata"]["generation_provider"] == "openai_responses"
    assert latest_trace["metadata"]["generation_result_status"] == "skipped"
    assert latest_trace["metadata"]["answer_generated"] is False
    assert latest_trace["metadata"]["skip_reason"] == "no_retrieved_context"
    assert "super-secret-key" not in captured.out
    assert "super-secret-key" not in json.dumps(latest_trace)


def test_ask_command_openai_missing_api_key_keeps_old_trace(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")
    (workspace_dir / "config.toml").write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = ""\n'
        ),
        encoding="utf-8",
    )

    exit_code = main(["ask", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Ask failed: Invalid config file: generation.api_key is required" in (
        captured.out
    )
    assert "generation.provider is openai_responses" in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace


def test_ask_command_openai_provider_failure_keeps_old_trace(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    latest_trace_path = workspace_dir / "traces" / "latest_trace.json"
    ingest_trace = latest_trace_path.read_text(encoding="utf-8")
    (workspace_dir / "config.toml").write_text(
        (
            "[generation]\n"
            'provider = "openai_responses"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "gpt-4o-mini"\n'
            'api_key = "super-secret-key"\n'
        ),
        encoding="utf-8",
    )

    def fake_post(url, *, headers, json, timeout):
        raise RuntimeError("boom super-secret-key")

    monkeypatch.setattr(
        "ragent_forge.app.services.generation_service.httpx.post",
        fake_post,
    )

    exit_code = main(["ask", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Ask failed: Generation provider failed: boom" in captured.out
    assert "super-secret-key" not in captured.out
    assert latest_trace_path.read_text(encoding="utf-8") == ingest_trace


def test_ask_command_prints_friendly_message_when_chunks_are_missing(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_dir = tmp_path / ".ragent"

    exit_code = main(["ask", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No chunks found. Run ragent ingest <path> first." in captured.out
    assert not (workspace_dir / "traces" / "latest_trace.json").exists()


def test_ask_command_reports_corrupt_chunks_json(tmp_path: Path, capsys) -> None:
    workspace_dir = tmp_path / ".ragent"
    chunks_dir = workspace_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("not-json\n", encoding="utf-8")

    exit_code = main(["ask", "agent", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Ask failed:" in captured.out
    assert "Invalid JSON in chunks file" in captured.out
    assert not (workspace_dir / "traces" / "latest_trace.json").exists()


def test_traces_latest_command_prints_latest_trace_after_ask(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("agent memory", encoding="utf-8")
    workspace_dir = tmp_path / ".ragent"
    assert main(["ingest", str(knowledge_dir), "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()
    assert main(["ask", "agent", "--workspace", str(workspace_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["traces", "latest", "--workspace", str(workspace_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Trace ID: ask-retrieval-" in captured.out
    assert "Operation: ask_retrieval" in captured.out
    assert "Status: success" in captured.out
    assert "1. read_chunks" in captured.out
    assert "2. retrieve_context" in captured.out
    assert "- question: agent" in captured.out
