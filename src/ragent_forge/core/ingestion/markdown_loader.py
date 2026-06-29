from __future__ import annotations

from pathlib import Path

from ragent_forge.app.models import Document

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_document(path: str | Path) -> Document:
    source_path = Path(path).expanduser()

    if not source_path.exists():
        raise FileNotFoundError(f"Document not found: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"Document path is not a file: {source_path}")

    extension = source_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{extension}' for {source_path}. "
            f"Supported file types: {supported}."
        )

    text = source_path.read_text(encoding="utf-8")
    resolved_path = source_path.resolve()
    metadata = {
        "source_path": str(resolved_path),
        "file_name": resolved_path.name,
        "extension": extension,
        "character_count": len(text),
    }
    return Document(id=str(resolved_path), text=text, metadata=metadata)
