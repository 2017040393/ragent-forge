from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from time import perf_counter

from ragent_forge.app.ports import ChunkReader
from ragent_forge.app.services.vector_index_service import VectorIndexRecord
from ragent_forge.core.retrieval.contracts import ChunkRecord


@dataclass(frozen=True)
class PreparedBM25Chunk:
    record: ChunkRecord
    text: str
    tokens: tuple[str, ...]
    term_frequencies: Counter[str]
    length: int


@dataclass(frozen=True)
class PreparedChunkState:
    snapshot_id: str | None
    records: tuple[ChunkRecord, ...]
    lexical_chunks: tuple[tuple[ChunkRecord, tuple[str, ...]], ...]
    bm25_chunks: tuple[PreparedBM25Chunk, ...]
    document_frequencies: Counter[str]
    average_document_length: float
    load_latency_ms: float


@dataclass(frozen=True)
class PreparedVectorState:
    snapshot_id: str | None
    records: tuple[VectorIndexRecord, ...]
    chunk_by_id: dict[str, ChunkRecord]
    load_latency_ms: float


@dataclass(frozen=True)
class PreparedCacheStats:
    snapshot_id: str | None
    chunk_loads: int
    vector_loads: int
    warm_hits: int
    invalidations: int
    last_chunk_load_latency_ms: float | None
    last_vector_load_latency_ms: float | None


class PreparedStateCache:
    """Snapshot-scoped prepared retrieval state shared by search adapters."""

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]],
    ) -> None:
        self._tokenizer = tokenizer
        self._lock = RLock()
        self._snapshot_id: str | None = None
        self._chunk_state: PreparedChunkState | None = None
        self._vector_state: PreparedVectorState | None = None
        self._chunk_loads = 0
        self._vector_loads = 0
        self._warm_hits = 0
        self._invalidations = 0
        self._last_chunk_latency_ms: float | None = None
        self._last_vector_latency_ms: float | None = None

    def prepare_chunks(self, workspace: ChunkReader) -> PreparedChunkState:
        snapshot_id = _snapshot_id(workspace)
        with self._lock:
            if (
                self._chunk_state is not None
                and self._chunk_state.snapshot_id == snapshot_id
            ):
                self._warm_hits += 1
                return self._chunk_state

            started = perf_counter()
            records = tuple(workspace.read_chunks())
            lexical_chunks = tuple(
                (
                    record,
                    tuple(self._tokenizer(str(record.get("text", "")))),
                )
                for record in records
            )
            bm25_chunks = tuple(
                PreparedBM25Chunk(
                    record=record,
                    text=str(record.get("text", "")),
                    tokens=tokens,
                    term_frequencies=Counter(tokens),
                    length=len(tokens),
                )
                for record, tokens in lexical_chunks
            )
            document_frequencies: Counter[str] = Counter()
            for chunk in bm25_chunks:
                document_frequencies.update(set(chunk.tokens))
            average_document_length = (
                sum(chunk.length for chunk in bm25_chunks) / len(bm25_chunks)
                if bm25_chunks
                else 0.0
            )
            latency_ms = (perf_counter() - started) * 1000
            if self._chunk_state is not None:
                self._invalidations += 1
            self._snapshot_id = snapshot_id
            self._vector_state = None
            self._chunk_loads += 1
            self._last_chunk_latency_ms = latency_ms
            self._chunk_state = PreparedChunkState(
                snapshot_id=snapshot_id,
                records=records,
                lexical_chunks=lexical_chunks,
                bm25_chunks=bm25_chunks,
                document_frequencies=document_frequencies,
                average_document_length=average_document_length,
                load_latency_ms=latency_ms,
            )
            return self._chunk_state

    def prepare_vectors(
        self,
        workspace: ChunkReader,
        loader: Callable[[], list[VectorIndexRecord]],
    ) -> PreparedVectorState:
        snapshot_id = _snapshot_id(workspace)
        with self._lock:
            if (
                self._vector_state is not None
                and self._vector_state.snapshot_id == snapshot_id
            ):
                self._warm_hits += 1
                return self._vector_state

            started = perf_counter()
            records = tuple(loader())
            chunk_state = self.prepare_chunks(workspace)
            latency_ms = (perf_counter() - started) * 1000
            self._vector_loads += 1
            self._last_vector_latency_ms = latency_ms
            self._vector_state = PreparedVectorState(
                snapshot_id=chunk_state.snapshot_id,
                records=records,
                chunk_by_id={
                    str(record.get("chunk_id", "")): record
                    for record in chunk_state.records
                },
                load_latency_ms=latency_ms,
            )
            return self._vector_state

    def stats(self) -> PreparedCacheStats:
        with self._lock:
            return PreparedCacheStats(
                snapshot_id=self._snapshot_id,
                chunk_loads=self._chunk_loads,
                vector_loads=self._vector_loads,
                warm_hits=self._warm_hits,
                invalidations=self._invalidations,
                last_chunk_load_latency_ms=self._last_chunk_latency_ms,
                last_vector_load_latency_ms=self._last_vector_latency_ms,
            )


def _snapshot_id(workspace: object) -> str | None:
    current_snapshot = getattr(workspace, "current_snapshot_id", None)
    if not callable(current_snapshot):
        return None
    value = current_snapshot()
    return value if isinstance(value, str) else None
