import json
from pathlib import Path

from ragent_forge.cli import main


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


def test_ingest_command_prints_statistics(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.md").write_text("abcdefghij", encoding="utf-8")
    (knowledge_dir / "notes.txt").write_text("klmnopqrst", encoding="utf-8")
    (knowledge_dir / "skip.pdf").write_text("ignored", encoding="utf-8")

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
