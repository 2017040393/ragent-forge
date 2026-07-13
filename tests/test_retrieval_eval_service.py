import json
from pathlib import Path

import pytest

from ragent_forge.app.services import retrieval_eval_service
from ragent_forge.app.services.evidence_span_service import EvidenceSpan
from ragent_forge.app.services.retrieval_eval_service import (
    RetrievalEvalCase,
    RetrievalEvalService,
)
from ragent_forge.app.services.search_service import SearchResult
from ragent_forge.core.retrieval.contracts import (
    RetrievalRun,
    RetrievalStageRecord,
)


class FakeSearchService:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self.calls.append((query, limit))
        return self.results_by_query.get(query, [])[:limit]


class FakeWorkspace:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self.chunks = chunks
        self.calls = 0

    def read_chunks(self) -> list[dict[str, object]]:
        self.calls += 1
        return self.chunks


def make_result(
    chunk_id: str,
    source_path: str,
    score: float = 1.0,
    text: str = "full chunk text must stay out",
    *,
    start_char: int | None = 0,
    end_char: int | None = None,
    metadata: dict[str, object] | None = None,
) -> SearchResult:
    result_metadata: dict[str, object] = {
        "api_key": "embedding-secret-key",
        "embedding": [0.1, 0.2],
    }
    if metadata is not None:
        result_metadata.update(metadata)
    return SearchResult(
        chunk_id=chunk_id,
        document_id=source_path,
        source_path=source_path,
        start_char=start_char,
        end_char=len(text) if end_char is None else end_char,
        score=score,
        text=text,
        metadata=result_metadata,
    )


def make_evidence_span(
    span_id: str = "docs/rag.md::span-0001",
    *,
    source_path: str = "docs/rag.md",
    start_char: int | None = 10,
    end_char: int | None = 90,
    page_start: int | None = None,
    page_end: int | None = None,
) -> EvidenceSpan:
    return EvidenceSpan(
        id=span_id,
        source_path=source_path,
        document_id=source_path,
        start_char=start_char,
        end_char=end_char,
        text="Hybrid retrieval combines lexical and semantic retrieval.",
        media_type=(
            "application/pdf"
            if source_path.endswith(".pdf")
            else "text/markdown"
        ),
        section_title="Hybrid Retrieval",
        heading_path=("RAG", "Hybrid Retrieval"),
        block_types=("paragraph",),
        page_start=page_start,
        page_end=page_end,
        metadata={"text_sha256": "abc123"},
    )


def test_retrieval_eval_uses_engine_run_and_preserves_stage_trace() -> None:
    class FakeRetrievalEngine:
        def search(self, query: str, limit: int) -> list[SearchResult]:
            raise AssertionError("evaluate must call run(), not search()")

        def run(self, query: str, limit: int) -> RetrievalRun:
            result = make_result("docs/rag.md::chunk-0000", "docs/rag.md")
            return RetrievalRun(
                query=query,
                retrieval_mode="bm25",
                retrieval_method="bm25",
                requested_limit=limit,
                candidate_count=1,
                result_count=1,
                result_chunk_ids=[result.chunk_id],
                results=[result],
                stages=[
                    RetrievalStageRecord(
                        name="candidate_retrieval",
                        status="completed",
                        outputs={"candidate_count": 1},
                        latency_ms=2.5,
                    )
                ],
            )

    report = RetrievalEvalService().evaluate(
        cases=[
            RetrievalEvalCase(
                id="case-1",
                query="agent memory",
                expected_chunk_ids=["docs/rag.md::chunk-0000"],
            )
        ],
        search_service=FakeRetrievalEngine(),
        limit=5,
        retrieval_mode="bm25",
        retrieval_method="bm25",
        cases_path="cases.jsonl",
        workspace_path=".ragent",
    )

    assert report.passed_count == 1
    assert report.retrieval_pipeline[0]["name"] == "candidate_retrieval"
    assert report.results[0].metadata["retrieval_pipeline"][0]["status"] == (
        "completed"
    )
    assert report.stage_latency_ms["candidate_retrieval"].model_dump() == {
        "sample_count": 1,
        "average_ms": 2.5,
        "p50_ms": 2.5,
        "p95_ms": 2.5,
    }


def make_chunk_record(
    chunk_id: str,
    *,
    source_path: str = "docs/rag.md",
    start_char: int | None = 0,
    end_char: int | None = 120,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": source_path,
        "source_path": source_path,
        "start_char": start_char,
        "end_char": end_char,
        "metadata": metadata or {"source_path": source_path},
        "text": "current chunk text",
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_cases_validates_deduplicates_and_preserves_unknown_metadata(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "retrieval_cases.jsonl"
    write_jsonl(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "agent memory",
                "expected_chunk_ids": [
                    "rag.md::chunk-0000",
                    "rag.md::chunk-0000",
                ],
                "expected_source_paths": ["rag.md", "rag.md"],
                "difficulty": "easy",
                "notes": "human note",
            }
        ],
    )

    cases = RetrievalEvalService().load_cases(cases_path)

    assert len(cases) == 1
    assert cases[0].id == "case-001"
    assert cases[0].expected_chunk_ids == ["rag.md::chunk-0000"]
    assert cases[0].expected_source_paths == ["rag.md"]
    assert cases[0].metadata == {"difficulty": "easy"}
    assert cases[0].notes == "human note"


def test_load_cases_accepts_evidence_spans_only_case(tmp_path: Path) -> None:
    cases_path = tmp_path / "retrieval_cases.jsonl"
    write_jsonl(
        cases_path,
        [
            {
                "id": "case-001",
                "query": "Why does the system use hybrid retrieval?",
                "evidence_spans": [
                    {
                        "id": "docs/rag.md::span-0001",
                        "source_path": "docs/rag.md",
                        "document_id": "docs/rag.md",
                        "start_char": 100,
                        "end_char": 500,
                        "text": (
                            "Hybrid retrieval combines lexical and semantic "
                            "retrieval."
                        ),
                        "media_type": "text/markdown",
                        "section_title": "Hybrid Retrieval",
                        "heading_path": ["RAG", "Hybrid Retrieval"],
                        "block_types": ["paragraph"],
                        "page_start": None,
                        "page_end": None,
                        "metadata": {"text_sha256": "abc123"},
                    }
                ],
                "reference_answer": (
                    "Hybrid retrieval combines keyword matching and semantic "
                    "matching."
                ),
            }
        ],
    )

    cases = RetrievalEvalService().load_cases(cases_path)

    assert len(cases) == 1
    case = cases[0]
    assert case.expected_chunk_ids == []
    assert case.expected_source_paths == []
    assert len(case.evidence_spans) == 1
    span = case.evidence_spans[0]
    assert span.id == "docs/rag.md::span-0001"
    assert span.source_path == "docs/rag.md"
    assert span.heading_path == ("RAG", "Hybrid Retrieval")
    assert span.block_types == ("paragraph",)
    assert "evidence_spans" not in case.metadata
    assert case.metadata["reference_answer"] == (
        "Hybrid retrieval combines keyword matching and semantic matching."
    )


def test_load_cases_rejects_invalid_json_with_line_number(tmp_path: Path) -> None:
    cases_path = tmp_path / "retrieval_cases.jsonl"
    cases_path.write_text(
        (
            '{"id": "ok", "query": "agent", '
            '"expected_source_paths": ["rag.md"]}\n'
            "{not-json}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        RetrievalEvalService().load_cases(cases_path)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {"query": "agent", "expected_source_paths": ["rag.md"]},
            "id",
        ),
        (
            {"id": "case-001", "expected_source_paths": ["rag.md"]},
            "query",
        ),
        (
            {"id": "case-001", "query": "agent"},
            "expected_chunk_ids, expected_source_paths, or evidence_spans",
        ),
    ],
)
def test_load_cases_rejects_invalid_case_shape_with_line_number(
    tmp_path: Path,
    record: dict[str, object],
    message: str,
) -> None:
    cases_path = tmp_path / "retrieval_cases.jsonl"
    write_jsonl(cases_path, [record])

    with pytest.raises(ValueError, match=f"line 1.*{message}"):
        RetrievalEvalService().load_cases(cases_path)


def test_load_cases_rejects_empty_files(tmp_path: Path) -> None:
    cases_path = tmp_path / "retrieval_cases.jsonl"
    cases_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no eval cases found"):
        RetrievalEvalService().load_cases(cases_path)


def test_evaluate_matches_chunk_id_before_source_path() -> None:
    service = RetrievalEvalService()
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="agent",
            expected_chunk_ids=["wanted::chunk-0002"],
            expected_source_paths=["early.md"],
        )
    ]
    search = FakeSearchService(
        {
            "agent": [
                make_result("early::chunk-0000", "early.md", 2.0),
                make_result("wanted::chunk-0002", "wanted.md", 1.0),
            ]
        }
    )

    report = service.evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    result = report.results[0]
    assert result.passed is True
    assert result.rank == 2
    assert result.matched_by == "chunk_id"
    assert result.reciprocal_rank == pytest.approx(0.5)
    assert result.failure_type is None
    assert result.failure_reason is None


def test_evaluate_maps_evidence_spans_to_current_chunks_and_passes_retrieval() -> None:
    span = make_evidence_span(start_char=20, end_char=80)
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="hybrid retrieval",
            evidence_spans=[span],
        )
    ]
    workspace = FakeWorkspace(
        [make_chunk_record("docs/rag.md::chunk-0000", start_char=0, end_char=100)]
    )
    search = FakeSearchService(
        {
            "hybrid retrieval": [
                make_result(
                    "docs/rag.md::chunk-0000",
                    "docs/rag.md",
                    text="x" * 30,
                    start_char=20,
                    end_char=50,
                )
            ]
        }
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
        workspace=workspace,
    )

    result = report.results[0]
    assert result.passed is True
    assert result.matched_by == "chunk_id"
    assert result.expected_chunk_ids == ["docs/rag.md::chunk-0000"]
    assert result.metadata["evidence_span_count"] == 1
    assert result.metadata["mapped_expected_chunk_ids"] == [
        "docs/rag.md::chunk-0000"
    ]
    assert result.metadata["unmatched_span_ids"] == []
    assert result.mapping_coverage == pytest.approx(1.0)
    assert result.evidence_coverage == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.2)
    assert result.ndcg == pytest.approx(1.0)
    assert result.context_evidence_density == pytest.approx(1.0)
    assert result.duplicate_context_ratio == pytest.approx(0.0)
    assert report.metrics["mapping_coverage"] == pytest.approx(1.0)
    assert report.metrics["evidence_coverage@k"] == pytest.approx(0.5)
    assert workspace.calls == 1


def test_evaluate_fails_when_evidence_spans_cannot_be_mapped() -> None:
    span = make_evidence_span(
        span_id="docs/missing.md::span-0001",
        source_path="docs/missing.md",
        start_char=20,
        end_char=80,
    )
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="missing span",
            evidence_spans=[span],
        )
    ]
    workspace = FakeWorkspace(
        [make_chunk_record("docs/rag.md::chunk-0000", start_char=0, end_char=100)]
    )
    search = FakeSearchService(
        {"missing span": [make_result("docs/rag.md::chunk-0000", "docs/rag.md")]}
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
        workspace=workspace,
    )

    result = report.results[0]
    assert result.passed is False
    assert result.matched_by == "none"
    assert result.expected_chunk_ids == []
    assert result.metadata["mapped_expected_chunk_ids"] == []
    assert result.metadata["unmatched_span_ids"] == ["docs/missing.md::span-0001"]
    assert result.mapping_coverage == pytest.approx(0.0)
    assert result.evidence_coverage == pytest.approx(0.0)
    assert result.failure_type == "unmapped_evidence"
    assert result.failure_reason == (
        "Evidence spans could not be mapped to current chunks."
    )


def test_evaluate_matches_source_path_when_no_chunk_match_exists() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="retrieval",
            expected_source_paths=["rag.md"],
        )
    ]
    search = FakeSearchService(
        {"retrieval": [make_result("other::chunk-0000", "rag.md", 1.0)]}
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=3,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    result = report.results[0]
    assert result.passed is True
    assert result.rank == 1
    assert result.matched_by == "source_path"
    assert result.actual_chunk_ids == ["other::chunk-0000"]
    assert result.actual_source_paths == ["rag.md"]
    assert result.relevant_result_ranks == [1]
    assert result.precision == pytest.approx(0.3333)
    assert result.ndcg == pytest.approx(1.0)


def test_evaluate_combines_manual_expected_chunk_ids_and_mapped_spans() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="hybrid retrieval",
            expected_chunk_ids=["manual::chunk-0001"],
            evidence_spans=[
                make_evidence_span(
                    span_id="docs/rag.md::span-0001",
                    start_char=20,
                    end_char=80,
                )
            ],
        )
    ]
    workspace = FakeWorkspace(
        [make_chunk_record("mapped::chunk-0002", start_char=0, end_char=100)]
    )
    search = FakeSearchService(
        {"hybrid retrieval": [make_result("mapped::chunk-0002", "docs/rag.md")]}
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
        workspace=workspace,
    )

    result = report.results[0]
    assert result.passed is True
    assert result.matched_by == "chunk_id"
    assert result.expected_chunk_ids == ["manual::chunk-0001", "mapped::chunk-0002"]
    assert result.metadata["mapped_expected_chunk_ids"] == ["mapped::chunk-0002"]


def test_evaluate_maps_pdf_evidence_span_by_page_overlap() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="table evidence",
            evidence_spans=[
                make_evidence_span(
                    span_id="docs/paper.pdf::span-0001",
                    source_path="docs/paper.pdf",
                    start_char=None,
                    end_char=None,
                    page_start=3,
                    page_end=4,
                )
            ],
        )
    ]
    workspace = FakeWorkspace(
        [
            make_chunk_record(
                "docs/paper.pdf::chunk-0000",
                source_path="docs/paper.pdf",
                start_char=None,
                end_char=None,
                metadata={"source_path": "docs/paper.pdf", "page_number": 4},
            )
        ]
    )
    search = FakeSearchService(
        {
            "table evidence": [
                make_result(
                    "docs/paper.pdf::chunk-0000",
                    "docs/paper.pdf",
                    start_char=None,
                    end_char=None,
                    metadata={"page_number": 4},
                )
            ]
        }
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
        workspace=workspace,
    )

    result = report.results[0]
    assert result.passed is True
    assert result.matched_by == "chunk_id"
    assert result.expected_chunk_ids == ["docs/paper.pdf::chunk-0000"]
    assert result.metadata["mapped_expected_chunk_ids"] == [
        "docs/paper.pdf::chunk-0000"
    ]
    assert result.mapping_coverage == pytest.approx(1.0)
    assert result.evidence_coverage == pytest.approx(0.5)


def test_evaluate_matches_repo_relative_source_path_against_absolute_result() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="retrieval",
            expected_source_paths=["examples/knowledge/rag_basics.md"],
        )
    ]
    search = FakeSearchService(
        {
            "retrieval": [
                make_result(
                    "C:\\repo\\ragent-forge\\examples\\knowledge\\rag_basics.md::chunk-0000",
                    "C:\\repo\\ragent-forge\\examples\\knowledge\\rag_basics.md",
                    1.0,
                )
            ]
        }
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=3,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("examples/eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    result = report.results[0]
    assert result.passed is True
    assert result.rank == 1
    assert result.matched_by == "source_path"


def test_evaluate_records_failed_cases_and_empty_retrieval_results() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="missing",
            expected_chunk_ids=["wanted::chunk-0000"],
        )
    ]
    search = FakeSearchService({"missing": []})

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    result = report.results[0]
    assert result.passed is False
    assert result.rank is None
    assert result.reciprocal_rank == 0.0
    assert result.matched_by == "none"
    assert result.actual_chunk_ids == []
    assert result.actual_source_paths == []
    assert result.top_results == []
    assert result.failure_type == "no_result"
    assert result.failure_reason == "No retrieval results returned."


def test_evaluate_classifies_failed_case_when_expected_source_is_missed() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="missing source",
            expected_source_paths=["docs/expected.md"],
        )
    ]
    search = FakeSearchService(
        {"missing source": [make_result("other::chunk-0000", "docs/other.md")]}
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    result = report.results[0]
    assert result.passed is False
    assert result.failure_type == "missed_source"
    assert (
        result.failure_reason
        == "Retrieved results did not include any expected source path."
    )


def test_evaluate_classifies_wrong_section_for_mapped_span_source_match() -> None:
    span = make_evidence_span(start_char=20, end_char=80)
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="wrong section",
            evidence_spans=[span],
        )
    ]
    workspace = FakeWorkspace(
        [
            make_chunk_record(
                "docs/rag.md::chunk-0000",
                source_path="docs/rag.md",
                start_char=0,
                end_char=100,
            )
        ]
    )
    search = FakeSearchService(
        {
            "wrong section": [
                make_result("docs/rag.md::chunk-0009", "docs/rag.md")
            ]
        }
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
        workspace=workspace,
    )

    result = report.results[0]
    assert result.passed is False
    assert result.expected_chunk_ids == ["docs/rag.md::chunk-0000"]
    assert result.failure_type == "wrong_section"
    assert (
        result.failure_reason
        == "Expected source was retrieved, but no expected chunk was found in top-k."
    )


def test_evaluate_classifies_low_rank_when_expected_chunks_are_missing() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="low rank",
            expected_chunk_ids=["docs/rag.md::chunk-0000"],
        )
    ]
    search = FakeSearchService(
        {"low rank": [make_result("docs/other.md::chunk-0000", "docs/other.md")]}
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    result = report.results[0]
    assert result.passed is False
    assert result.failure_type == "low_rank"
    assert (
        result.failure_reason
        == "Expected chunks were not found within the evaluated top-k results."
    )


def test_classify_failure_falls_back_to_unknown_for_unmatched_shape() -> None:
    case = RetrievalEvalCase(
        id="case-001",
        query="ambiguous",
        evidence_spans=[make_evidence_span()],
    )

    failure_type, failure_reason = retrieval_eval_service._classify_failure(
        case=case,
        expected_chunk_ids=[],
        search_results=[make_result("other::chunk-0000", "docs/rag.md")],
        matched_by="none",
        metadata={},
    )

    assert failure_type == "unknown"
    assert failure_reason == "No deterministic failure heuristic matched."


def test_evaluate_computes_hit_rates_and_mrr() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="one",
            expected_chunk_ids=["a::chunk-0000"],
        ),
        RetrievalEvalCase(
            id="case-002",
            query="three",
            expected_chunk_ids=["c::chunk-0002"],
        ),
        RetrievalEvalCase(
            id="case-003",
            query="miss",
            expected_chunk_ids=["missing::chunk-0000"],
        ),
    ]
    search = FakeSearchService(
        {
            "one": [make_result("a::chunk-0000", "a.md")],
            "three": [
                make_result("a::chunk-0000", "a.md"),
                make_result("b::chunk-0001", "b.md"),
                make_result("c::chunk-0002", "c.md"),
            ],
            "miss": [make_result("other::chunk-0000", "other.md")],
        }
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=3,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    assert report.case_count == 3
    assert report.passed_count == 2
    assert report.failed_count == 1
    assert report.metrics["hit@1"] == pytest.approx(0.3333)
    assert report.metrics["hit@3"] == pytest.approx(0.6667)
    assert report.metrics["hit@5"] == pytest.approx(0.6667)
    assert report.metrics["hit@k"] == pytest.approx(0.6667)
    assert report.metrics["mrr"] == pytest.approx(0.4444)


def test_evaluate_computes_recall_latency_and_context_metrics() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="partial",
            expected_chunk_ids=["a::chunk-0000", "b::chunk-0001"],
        ),
        RetrievalEvalCase(
            id="case-002",
            query="full",
            expected_chunk_ids=["c::chunk-0002"],
        ),
    ]
    search = FakeSearchService(
        {
            "partial": [
                make_result(
                    "a::chunk-0000",
                    "a.md",
                    text="abcdefghi",
                )
            ],
            "full": [
                make_result(
                    "x::chunk-0000",
                    "x.md",
                    text="wxyz",
                ),
                make_result(
                    "c::chunk-0002",
                    "c.md",
                    text="abcde",
                ),
            ],
        }
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    first = report.results[0]
    second = report.results[1]
    assert first.retrieved_count == 1
    assert first.expected_chunk_count == 2
    assert first.recall == pytest.approx(0.5)
    assert first.retrieval_latency_ms >= 0.0
    assert first.retrieved_context_chars == 9
    assert first.estimated_context_tokens == 3
    assert second.retrieved_count == 2
    assert second.expected_chunk_count == 1
    assert second.recall == pytest.approx(1.0)
    assert second.retrieval_latency_ms >= 0.0
    assert second.retrieved_context_chars == 9
    assert second.estimated_context_tokens == 3
    assert report.metrics["recall@k"] == pytest.approx(0.75)
    assert report.metrics["avg_retrieval_latency_ms"] >= 0.0
    assert report.metrics["avg_retrieved_count"] == pytest.approx(1.5)
    assert report.metrics["avg_retrieved_context_chars"] == pytest.approx(9.0)
    assert report.metrics["avg_estimated_context_tokens"] == pytest.approx(3.0)


def test_evaluate_computes_precision_ndcg_density_and_duplicate_metrics() -> None:
    shared_text = "shared context text repeated alpha"
    unique_text = "unique relevant beta"
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="ranked evidence",
            expected_chunk_ids=["a::chunk-0000", "c::chunk-0002"],
        )
    ]
    search = FakeSearchService(
        {
            "ranked evidence": [
                make_result("a::chunk-0000", "a.md", text=shared_text),
                make_result("x::chunk-0001", "x.md", text=shared_text),
                make_result("c::chunk-0002", "c.md", text=unique_text),
            ]
        }
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
    )

    result = report.results[0]
    assert result.relevant_retrieved_count == 2
    assert result.relevant_result_ranks == [1, 3]
    assert result.precision == pytest.approx(0.4)
    assert result.ndcg == pytest.approx(0.9197)
    assert result.context_evidence_density == pytest.approx(
        (len(shared_text) + len(unique_text))
        / (2 * len(shared_text) + len(unique_text)),
        abs=1e-4,
    )
    assert result.duplicate_context_ratio > 0.0
    assert report.metrics["precision@1"] == pytest.approx(1.0)
    assert report.metrics["precision@3"] == pytest.approx(0.6667)
    assert report.metrics["precision@5"] == pytest.approx(0.4)
    assert report.metrics["precision@k"] == pytest.approx(0.4)
    assert report.metrics["ndcg@k"] == pytest.approx(0.9197)
    assert report.metrics["mapping_coverage"] == pytest.approx(0.0)
    assert report.metrics["mapping_coverage_case_rate"] == pytest.approx(0.0)
    assert report.metrics["evidence_coverage@k"] == pytest.approx(0.0)
    assert report.metrics["evidence_coverage_case_rate"] == pytest.approx(0.0)
    assert report.metrics["retrieval_latency_p50_ms"] >= 0.0
    assert report.metrics["retrieval_latency_p95_ms"] >= 0.0


def test_evaluate_computes_partial_mapping_and_evidence_coverage() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="partial span coverage",
            evidence_spans=[
                make_evidence_span(
                    span_id="docs/rag.md::span-0001",
                    source_path="docs/rag.md",
                    start_char=0,
                    end_char=20,
                ),
                make_evidence_span(
                    span_id="docs/missing.md::span-0001",
                    source_path="docs/missing.md",
                    start_char=0,
                    end_char=20,
                ),
            ],
        )
    ]
    workspace = FakeWorkspace(
        [
            make_chunk_record(
                "docs/rag.md::chunk-0000",
                source_path="docs/rag.md",
                start_char=0,
                end_char=20,
            )
        ]
    )
    search = FakeSearchService(
        {
            "partial span coverage": [
                make_result(
                    "docs/rag.md::chunk-0000",
                    "docs/rag.md",
                    text="x" * 20,
                    start_char=0,
                    end_char=20,
                )
            ]
        }
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
        workspace=workspace,
    )

    result = report.results[0]
    assert result.mapping_coverage == pytest.approx(0.5)
    assert result.evidence_coverage == pytest.approx(0.5)
    assert report.metrics["mapping_coverage"] == pytest.approx(0.5)
    assert report.metrics["mapping_coverage_case_rate"] == pytest.approx(1.0)
    assert report.metrics["evidence_coverage@k"] == pytest.approx(0.5)
    assert report.metrics["evidence_coverage_case_rate"] == pytest.approx(1.0)


def test_percentile_uses_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert retrieval_eval_service._percentile(values, 0.5) == pytest.approx(2.5)
    assert retrieval_eval_service._percentile(values, 0.95) == pytest.approx(3.85)


def test_evaluate_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        RetrievalEvalService().evaluate(
            cases=[
                RetrievalEvalCase(
                    id="case-001",
                    query="agent",
                    expected_source_paths=["rag.md"],
                )
            ],
            search_service=FakeSearchService({}),
            limit=0,
            retrieval_mode="lexical",
            retrieval_method="lexical_token_overlap",
            cases_path=Path("eval/retrieval_cases.jsonl"),
            workspace_path=Path(".ragent"),
        )


def test_report_excludes_full_text_api_keys_and_embeddings() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="agent",
            expected_chunk_ids=["rag.md::chunk-0000"],
        )
    ]
    search = FakeSearchService(
        {"agent": [make_result("rag.md::chunk-0000", "rag.md")]}
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="semantic",
        retrieval_method="semantic_cosine_similarity",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        index_path=Path(".ragent/index/vector_index.jsonl"),
    )

    report_json = report.model_dump_json(exclude_none=True)
    assert "full chunk text must stay out" not in report_json
    assert "embedding-secret-key" not in report_json
    assert '"embedding"' not in report_json
    assert report.embedding_provider == "openai_embeddings"
    assert report.embedding_model == "text-embedding-3-small"
    assert report.index_path == str(Path(".ragent/index/vector_index.jsonl"))


def test_hybrid_report_records_fusion_metadata() -> None:
    cases = [
        RetrievalEvalCase(
            id="case-001",
            query="agent",
            expected_chunk_ids=["rag.md::chunk-0000"],
        )
    ]
    search = FakeSearchService(
        {"agent": [make_result("rag.md::chunk-0000", "rag.md")]}
    )

    report = RetrievalEvalService().evaluate(
        cases=cases,
        search_service=search,
        limit=5,
        retrieval_mode="hybrid",
        retrieval_method="hybrid_rrf",
        cases_path=Path("eval/retrieval_cases.jsonl"),
        workspace_path=Path(".ragent"),
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        index_path=Path(".ragent/index/vector_index.jsonl"),
        fusion_method="reciprocal_rank_fusion",
        rrf_k=60,
        sparse_method="bm25",
        dense_method="semantic_cosine_similarity",
        sparse_weight=1.0,
        dense_weight=1.0,
        lexical_weight=1.0,
        semantic_weight=1.0,
    )

    report_json = report.model_dump_json(exclude_none=True)
    assert report.retrieval_mode == "hybrid"
    assert report.retrieval_method == "hybrid_rrf"
    assert report.fusion_method == "reciprocal_rank_fusion"
    assert report.rrf_k == 60
    assert report.sparse_method == "bm25"
    assert report.dense_method == "semantic_cosine_similarity"
    assert report.sparse_weight == 1.0
    assert report.dense_weight == 1.0
    assert report.lexical_weight == 1.0
    assert report.semantic_weight == 1.0
    assert "full chunk text must stay out" not in report_json
    assert "embedding-secret-key" not in report_json
    assert '"embedding"' not in report_json
