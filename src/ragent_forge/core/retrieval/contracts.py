from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from ragent_forge.core.models import (
    SourceAuthority,
    SourceKind,
    SourceLifecycle,
    SourceRef,
)
from ragent_forge.core.retrieval.types import RetrievalMethod, RetrievalMode


class RetrievalMetadata(dict[str, object]):
    """Validated JSON metadata with typed accessors for retrieval code."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: object,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.from_value,
            core_schema.dict_schema(
                core_schema.str_schema(),
                core_schema.any_schema(),
            ),
        )

    @classmethod
    def from_value(cls, value: object) -> RetrievalMetadata:
        if isinstance(value, RetrievalMetadata):
            return cls(value)
        if not isinstance(value, Mapping):
            raise ValueError("retrieval metadata must be an object")
        return cls(
            {
                str(key): _validated_json_value(item)
                for key, item in value.items()
            }
        )

    def string(self, key: str) -> str | None:
        value = self.get(key)
        return value if isinstance(value, str) else None

    def integer(self, key: str) -> int | None:
        value = self.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def number(self, key: str) -> float | None:
        value = self.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        return None

    def boolean(self, key: str) -> bool | None:
        value = self.get(key)
        return value if isinstance(value, bool) else None

    def strings(self, key: str) -> list[str]:
        value = self.get(key)
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    def to_dict(self) -> dict[str, object]:
        return dict(self)


MetadataRecord = RetrievalMetadata

SourceRecord = SourceRef
"""Canonical source identity contract shared by retrieval and answer views."""


class ChunkRecord(TypedDict):
    """JSON-compatible chunk record exchanged by workspace adapters."""

    schema_version: int
    snapshot_id: str | None
    chunk_id: str
    document_id: str
    text: str
    source_path: str | None
    start_char: int | None
    end_char: int | None
    metadata: dict[str, object]
    source_kind: SourceKind
    provenance: str | None
    authority: SourceAuthority
    freshness: str | None
    lifecycle: SourceLifecycle


class RetrievalCandidate(BaseModel):
    """A ranked chunk candidate produced by a retrieval adapter."""

    chunk_id: str
    document_id: str
    source_path: str
    start_char: int | None = None
    end_char: int | None = None
    score: float
    text: str
    metadata: MetadataRecord = Field(default_factory=MetadataRecord)
    source_kind: SourceKind = "document"
    provenance: str | None = None
    authority: SourceAuthority = "source"
    freshness: str | None = None
    lifecycle: SourceLifecycle = "regenerable"

    def __init__(
        self,
        *,
        chunk_id: str,
        document_id: str,
        source_path: str,
        score: float,
        text: str,
        start_char: int | None = None,
        end_char: int | None = None,
        metadata: Mapping[str, object] | None = None,
        source_kind: SourceKind = "document",
        provenance: str | None = None,
        authority: SourceAuthority = "source",
        freshness: str | None = None,
        lifecycle: SourceLifecycle = "regenerable",
    ) -> None:
        super().__init__(
            chunk_id=chunk_id,
            document_id=document_id,
            source_path=source_path,
            start_char=start_char,
            end_char=end_char,
            score=score,
            text=text,
            metadata=MetadataRecord.from_value(metadata or {}),
            source_kind=source_kind,
            provenance=provenance,
            authority=authority,
            freshness=freshness,
            lifecycle=lifecycle,
        )


RetrievalStageName = Literal[
    "normalize_query",
    "candidate_retrieval",
    "deduplicate",
    "rerank",
    "context_selection",
    "trace",
]
RetrievalStageStatus = Literal["completed", "skipped", "failed"]


class RetrievalStageRecord(BaseModel):
    name: RetrievalStageName
    status: RetrievalStageStatus
    inputs: dict[str, object] = Field(default_factory=dict)
    outputs: dict[str, object] = Field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None


class RetrievalRun(BaseModel):
    query: str
    retrieval_mode: RetrievalMode
    retrieval_method: RetrievalMethod
    requested_limit: int
    candidate_count: int = 0
    result_count: int = 0
    result_chunk_ids: list[str] = Field(default_factory=list)
    results: list[RetrievalCandidate] = Field(default_factory=list)
    stages: list[RetrievalStageRecord] = Field(default_factory=list)
    snapshot_id: str | None = None
    metadata: MetadataRecord = Field(default_factory=MetadataRecord)


def _validated_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _validated_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_validated_json_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise ValueError(
        f"retrieval metadata values must be JSON-compatible, got "
        f"{type(value).__name__}"
    )
