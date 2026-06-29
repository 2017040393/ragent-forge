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
