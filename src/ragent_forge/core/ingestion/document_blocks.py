from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BlockType = Literal[
    "paragraph",
    "heading",
    "table",
    "formula",
    "list",
    "code",
    "blockquote",
    "caption",
    "unknown",
]


@dataclass(frozen=True)
class DocumentBlock:
    source_path: str
    media_type: str
    page_number: int | None
    block_index: int
    block_type: BlockType
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PdfExtractionWarning:
    source_path: str
    page: int | None
    kind: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "page": self.page,
            "kind": self.kind,
            "message": self.message,
        }
