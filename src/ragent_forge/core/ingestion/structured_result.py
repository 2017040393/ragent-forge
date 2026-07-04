from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragent_forge.app.models import Document
from ragent_forge.core.ingestion.document_blocks import DocumentBlock


@dataclass(frozen=True)
class StructuredLoadResult:
    document: Document
    blocks: tuple[DocumentBlock, ...]
    metadata: dict[str, Any]
    warnings: tuple[Any, ...] = field(default_factory=tuple)
