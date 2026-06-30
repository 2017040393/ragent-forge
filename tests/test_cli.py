import json
from pathlib import Path

from ragent_forge.cli import main


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
    assert (workspace_dir / "chunks" / "chunks.jsonl").is_file()
    assert (workspace_dir / "ingest" / "latest_summary.json").is_file()

    summary = json.loads(
        (workspace_dir / "ingest" / "latest_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["document_count"] == 2
    assert summary["chunk_count"] == 4
    assert summary["skipped_count"] == 1


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
