import json
from pathlib import Path

from ragent_forge.app.workspace import LocalWorkspace
from ragent_forge.cli import main
from ragent_forge.tui.view_models import run_tui_search


def test_cli_and_tui_persist_the_same_retrieval_run_shape(
    tmp_path: Path,
    capsys,
) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "rag.md").write_text(
        "Agentic RAG uses planning before retrieval.",
        encoding="utf-8",
    )
    workspace_path = tmp_path / ".ragent"

    assert main(["ingest", str(knowledge), "--workspace", str(workspace_path)]) == 0
    capsys.readouterr()
    assert main(
        [
            "search",
            "planning",
            "--workspace",
            str(workspace_path),
            "--retrieval",
            "lexical",
            "--limit",
            "3",
        ]
    ) == 0
    capsys.readouterr()
    workspace = LocalWorkspace(workspace_path)
    cli_trace = json.loads(
        workspace.latest_trace_path.read_text(encoding="utf-8")
    )
    cli_run = cli_trace["metadata"]["retrieval_run"]

    tui_state = run_tui_search(workspace_path, "planning", "lexical", 3)
    tui_trace = json.loads(
        workspace.latest_trace_path.read_text(encoding="utf-8")
    )
    tui_run = tui_trace["metadata"]["retrieval_run"]

    assert tui_state.trace_id == tui_trace["trace_id"]
    assert cli_trace["operation"] == tui_trace["operation"] == "search"
    assert cli_run.keys() == tui_run.keys()
    for key in (
        "query",
        "retrieval_mode",
        "retrieval_method",
        "requested_limit",
        "result_chunk_ids",
        "snapshot_id",
    ):
        assert cli_run[key] == tui_run[key]
    assert [
        (stage["name"], stage["status"])
        for stage in cli_run["stages"]
    ] == [
        (stage["name"], stage["status"])
        for stage in tui_run["stages"]
    ]
