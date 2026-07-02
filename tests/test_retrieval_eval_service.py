import json
from pathlib import Path

import pytest

from ragent_forge.app.services.retrieval_eval_service import (
    RetrievalEvalCase,
    RetrievalEvalService,
)
from ragent_forge.app.services.search_service import SearchResult


class FakeSearchService:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self.calls.append((query, limit))
        return self.results_by_query.get(query, [])[:limit]


def make_result(
    chunk_id: str,
    source_path: str,
    score: float = 1.0,
    text: str = "full chunk text must stay out",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=source_path,
        source_path=source_path,
        start_char=0,
        end_char=len(text),
        score=score,
        text=text,
        metadata={
            "api_key": "embedding-secret-key",
            "embedding": [0.1, 0.2],
        },
    )


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
            "expected_chunk_ids or expected_source_paths",
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
        lexical_weight=1.0,
        semantic_weight=1.0,
    )

    report_json = report.model_dump_json(exclude_none=True)
    assert report.retrieval_mode == "hybrid"
    assert report.retrieval_method == "hybrid_rrf"
    assert report.fusion_method == "reciprocal_rank_fusion"
    assert report.rrf_k == 60
    assert report.lexical_weight == 1.0
    assert report.semantic_weight == 1.0
    assert "full chunk text must stay out" not in report_json
    assert "embedding-secret-key" not in report_json
    assert '"embedding"' not in report_json
