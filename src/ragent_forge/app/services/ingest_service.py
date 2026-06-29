from pathlib import Path

from ragent_forge.app.models import IngestResult
from ragent_forge.core.chunking.simple_chunker import SimpleChunker
from ragent_forge.core.ingestion.markdown_loader import (
    SUPPORTED_EXTENSIONS,
    load_document,
)


class IngestService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
        self.chunker = SimpleChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest(self, path: str | Path) -> IngestResult:
        ingest_path = Path(path).expanduser()

        if not ingest_path.exists():
            raise FileNotFoundError(f"Ingest path not found: {ingest_path}")

        candidate_files = self._candidate_files(ingest_path)
        supported_files = [
            file_path
            for file_path in candidate_files
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        skipped_files = [
            str(file_path.resolve())
            for file_path in candidate_files
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ]

        if not supported_files:
            raise ValueError(
                f"No supported Markdown/TXT files found under: {ingest_path}"
            )

        documents = [load_document(file_path) for file_path in supported_files]
        chunks = [
            chunk
            for document in documents
            for chunk in self.chunker.chunk(document)
        ]

        return IngestResult(
            source_path=str(ingest_path.resolve()),
            documents=documents,
            chunks=chunks,
            skipped_files=skipped_files,
            metadata={
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
            },
        )

    def _candidate_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]

        return sorted(
            (file_path for file_path in path.rglob("*") if file_path.is_file()),
            key=lambda file_path: (
                len(file_path.relative_to(path).parts),
                str(file_path.relative_to(path)).lower(),
            ),
        )
