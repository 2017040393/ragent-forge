import json

import pytest

from ragent_forge.core.retrieval.contracts import (
    MetadataRecord,
    RetrievalCandidate,
)
from ragent_forge.core.retrieval.types import normalize_retrieval_mode


def test_retrieval_metadata_is_validated_and_json_serializable() -> None:
    metadata = MetadataRecord.from_value(
        {
            "retrieval_method": "bm25",
            "page_number": 3,
            "matched_modes": ["bm25", "semantic"],
        }
    )

    assert metadata.string("retrieval_method") == "bm25"
    assert metadata.integer("page_number") == 3
    assert metadata.strings("matched_modes") == ["bm25", "semantic"]
    assert json.loads(json.dumps(metadata)) == dict(metadata)


def test_retrieval_metadata_rejects_non_json_values() -> None:
    with pytest.raises(ValueError, match="must be JSON-compatible"):
        MetadataRecord.from_value({"invalid": object()})


def test_retrieval_candidate_carries_typed_source_provenance() -> None:
    candidate = RetrievalCandidate(
        chunk_id="fact-1",
        document_id="project-memory",
        source_path="project-memory",
        score=1.0,
        text="The project uses a local-first workspace.",
        source_kind="project_fact",
        provenance="user-confirmed",
        authority="user",
        freshness="2026-07-13T00:00:00Z",
        lifecycle="user_owned",
    )

    assert candidate.source_kind == "project_fact"
    assert candidate.provenance == "user-confirmed"
    assert candidate.authority == "user"
    assert candidate.lifecycle == "user_owned"


def test_retrieval_mode_normalization_is_strict() -> None:
    assert normalize_retrieval_mode(" BM25 ") == "bm25"

    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        normalize_retrieval_mode("bm-25")
