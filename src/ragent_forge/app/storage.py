from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

_WRITE_LOCK = RLock()


@contextmanager
def workspace_write_lock() -> Iterator[None]:
    """Serialize related workspace writes within the current process."""
    with _WRITE_LOCK:
        yield


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Write UTF-8 text without exposing a partially written destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    with _WRITE_LOCK:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    return destination
