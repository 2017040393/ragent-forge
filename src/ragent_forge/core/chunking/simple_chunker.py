from __future__ import annotations

from ragent_forge.app.models import Document, DocumentChunk


class SimpleChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        if not document.text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        source_path = str(document.metadata.get("source_path", document.id))

        while start < len(document.text):
            end = start + self.chunk_size
            chunk_index = len(chunks)
            chunks.append(
                DocumentChunk(
                    id=f"{source_path}::chunk-{chunk_index:04d}",
                    document_id=document.id,
                    text=document.text[start:end],
                    index=chunk_index,
                    metadata={
                        "source_path": source_path,
                        "start_char": start,
                        "end_char": min(end, len(document.text)),
                    },
                )
            )
            start += step

        return chunks
