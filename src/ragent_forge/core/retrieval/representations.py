from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Literal

from ragent_forge.core.retrieval.contracts import ChunkRecord

EmbeddingRepresentation = Literal[
    "raw_chunk_text_v1",
    "structured_document_text_v1",
]

EMBEDDING_REPRESENTATIONS: tuple[EmbeddingRepresentation, ...] = (
    "raw_chunk_text_v1",
    "structured_document_text_v1",
)

QueryEmbeddingRepresentation = Literal[
    "raw_query_v1",
    "instructed_query_v1",
]

QUERY_EMBEDDING_REPRESENTATIONS: tuple[QueryEmbeddingRepresentation, ...] = (
    "raw_query_v1",
    "instructed_query_v1",
)

INSTRUCTED_QUERY_V1_PREFIX = (
    "Instruct: Retrieve the document passage that best answers the query. "
    "Distinguish the correct section from other passages in the same source "
    "document.\nQuery: "
)


def build_embedding_text(
    chunk: ChunkRecord,
    representation: EmbeddingRepresentation = "raw_chunk_text_v1",
) -> str:
    """Build the deterministic document-side input for an embedding provider."""

    if representation == "raw_chunk_text_v1":
        return chunk["text"]
    if representation == "structured_document_text_v1":
        return build_structured_document_text_v1(chunk)
    raise ValueError(f"Unsupported embedding representation: {representation!r}")


def build_query_embedding_text(
    query: str,
    representation: QueryEmbeddingRepresentation = "raw_query_v1",
) -> str:
    """Build the deterministic query-side input for an embedding provider."""

    if representation == "raw_query_v1":
        return query
    if representation == "instructed_query_v1":
        return f"{INSTRUCTED_QUERY_V1_PREFIX}{query}"
    raise ValueError(
        f"Unsupported query embedding representation: {representation!r}"
    )


def build_structured_document_text_v1(chunk: ChunkRecord) -> str:
    metadata = chunk.get("metadata", {})
    block_types = _sorted_metadata_strings(
        metadata.get("block_types"), metadata.get("block_type")
    )
    heading_path = _metadata_strings(metadata.get("heading_path"))
    section = " > ".join(heading_path)
    if not section:
        section = _metadata_string(metadata.get("section_title")) or "unknown"

    title = (
        _metadata_string(metadata.get("document_title"))
        or _metadata_string(metadata.get("title"))
        or (heading_path[0] if heading_path else "")
        or _source_stem(chunk)
        or "unknown"
    )
    source = _stable_source(chunk)
    page = _page_label(metadata)
    signals = _signals(metadata, block_types)
    block_label = ", ".join(block_types) if block_types else "unknown"
    return "\n".join(
        (
            f"Document title: {title}",
            f"Source: {source}",
            f"Section: {section}",
            f"Page: {page}",
            f"Block types: {block_label}",
            f"Signals: {signals}",
            "Content:",
            chunk["text"],
        )
    )


def embedding_input_fingerprint(
    records: list[tuple[str, str]],
) -> str:
    """Hash ordered-independent chunk IDs and hashes for an index build."""

    payload = [
        {"chunk_id": chunk_id, "text_sha256": text_hash}
        for chunk_id, text_hash in sorted(records)
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_embedding_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_source(chunk: ChunkRecord) -> str:
    raw = _metadata_string(chunk.get("source_path")) or _metadata_string(
        chunk.get("document_id")
    )
    if not raw:
        return "unknown"
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        return PurePosixPath(normalized).name or "unknown"
    return normalized.lstrip("./") or "unknown"


def _source_stem(chunk: ChunkRecord) -> str:
    source = _stable_source(chunk)
    name = PurePosixPath(source).name
    if not name:
        return ""
    return PurePosixPath(name).stem


def _page_label(metadata: dict[str, object]) -> str:
    start = _metadata_integer(metadata.get("page_start"))
    end = _metadata_integer(metadata.get("page_end"))
    if start is None:
        return "unknown"
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def _signals(metadata: dict[str, object], block_types: list[str]) -> str:
    signals: list[str] = []
    if metadata.get("possible_formula") is True or any(
        "formula" in block_type.lower() for block_type in block_types
    ):
        signals.append("formula")
    if any("table" in block_type.lower() for block_type in block_types):
        signals.append("table")
    return ", ".join(signals) if signals else "none"


def _sorted_metadata_strings(*values: object) -> list[str]:
    return sorted(set(item for value in values for item in _metadata_strings(value)))


def _metadata_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list | tuple):
        return [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
    return []


def _metadata_string(value: object) -> str | None:
    values = _metadata_strings(value)
    return values[0] if values else None


def _metadata_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
