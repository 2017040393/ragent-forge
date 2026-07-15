from __future__ import annotations

import subprocess
from pathlib import Path

from benchmarks.context_selection_screen import (
    load_manifest,
    run_context_selection_screen,
)
from benchmarks.retrieval_baseline import (
    BaselineGitState,
    collect_runtime_environment,
)


def test_e4a_replays_frozen_e3b_top5_without_embedding_calls(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[1]
    manifest_path = (
        repository_root / "benchmarks/context_selection_screen_manifest_e4a.json"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()

    report = run_context_selection_screen(
        load_manifest(manifest_path),
        manifest_path=manifest_path,
        repository_root=repository_root,
        output_dir=tmp_path / "e4a-screen",
        git_state=BaselineGitState(
            commit=commit,
            branch="main",
            dirty=False,
        ),
        runtime_environment=collect_runtime_environment(),
    )

    assert report.valid is True
    assert report.promoted is True
    assert report.hybrid_top5_token_ratio == 0.9683
    assert all(gate.passed for gate in report.gates)
    assert [
        configuration.retrieval_mode
        for configuration in report.configurations
    ] == ["semantic", "hybrid"]
    for configuration in report.configurations:
        assert configuration.metrics.lost_hit_case_ids == []
        assert all(
            case.ranked_prefix_preserved for case in configuration.cases
        )
