from pathlib import Path


class IngestService:
    def ingest(self, path: str | Path) -> str:
        return (
            "Local Markdown/TXT ingestion will be implemented in v0.1. "
            f"Received path: {Path(path)}"
        )
