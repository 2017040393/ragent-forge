import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ragent_forge.app.storage import atomic_write_text


def test_atomic_write_keeps_existing_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "state.json"
    destination.write_text('{"state": "old"}\n', encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("ragent_forge.app.storage.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(destination, '{"state": "new"}\n')

    assert json.loads(destination.read_text(encoding="utf-8")) == {"state": "old"}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_concurrent_atomic_writes_leave_complete_json(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    payloads = [json.dumps({"writer": index}) + "\n" for index in range(20)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda payload: atomic_write_text(destination, payload),
                payloads,
            )
        )

    assert json.loads(destination.read_text(encoding="utf-8")) in [
        {"writer": index} for index in range(20)
    ]
    assert list(tmp_path.glob(".state.json.*.tmp")) == []
