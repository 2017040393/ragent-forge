from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ragent_forge.app.services.search_service import SearchResult


class SearchServiceProtocol(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]:
        ...


class RetrievalEvalCase(BaseModel):
    id: str
    query: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_source_paths: list[str] = Field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "query")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("expected_chunk_ids", "expected_source_paths")
    @classmethod
    def _dedupe_expected_values(cls, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if not normalized:
                raise ValueError("must contain non-empty strings")
            if normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped

    @model_validator(mode="after")
    def _requires_at_least_one_expected_value(self) -> Self:
        if not self.expected_chunk_ids and not self.expected_source_paths:
            raise ValueError(
                "expected_chunk_ids or expected_source_paths must be provided"
            )
        return self


class RetrievalEvalCaseResult(BaseModel):
    id: str
    query: str
    passed: bool
    rank: int | None = None
    reciprocal_rank: float = 0.0
    matched_by: Literal["chunk_id", "source_path", "none"]
    expected_chunk_ids: list[str]
    expected_source_paths: list[str]
    actual_chunk_ids: list[str]
    actual_source_paths: list[str]
    top_results: list[dict[str, Any]]


class RetrievalEvalReport(BaseModel):
    evaluation_type: Literal["retrieval"] = "retrieval"
    retrieval_mode: Literal["lexical", "semantic", "hybrid"]
    retrieval_method: str
    limit: int
    case_count: int
    passed_count: int
    failed_count: int
    metrics: dict[str, float]
    cases_path: str
    workspace: str
    results: list[RetrievalEvalCaseResult]
    embedding_provider: str | None = None
    embedding_model: str | None = None
    index_path: str | None = None
    fusion_method: str | None = None
    rrf_k: int | None = None
    lexical_weight: float | None = None
    semantic_weight: float | None = None


class RetrievalEvalService:
    def load_cases(self, cases_path: str | Path) -> list[RetrievalEvalCase]:
        path = Path(cases_path)
        if not path.is_file():
            raise FileNotFoundError(f"cases file not found: {path}")

        cases: list[RetrievalEvalCase] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in eval cases {path} at line "
                    f"{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid eval case at line {line_number}: expected object"
                )
            cases.append(self._case_from_payload(payload, line_number))

        if not cases:
            raise ValueError("no eval cases found")
        return cases

    def evaluate(
        self,
        *,
        cases: list[RetrievalEvalCase],
        search_service: SearchServiceProtocol,
        limit: int,
        retrieval_mode: Literal["lexical", "semantic", "hybrid"],
        retrieval_method: str,
        cases_path: str | Path,
        workspace_path: str | Path,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        index_path: str | Path | None = None,
        fusion_method: str | None = None,
        rrf_k: int | None = None,
        lexical_weight: float | None = None,
        semantic_weight: float | None = None,
    ) -> RetrievalEvalReport:
        if limit < 1:
            raise ValueError("limit must be greater than 0")
        if not cases:
            raise ValueError("no eval cases found")

        results = [
            self._evaluate_case(
                case=case,
                search_results=search_service.search(case.query, limit),
            )
            for case in cases
        ]
        passed_count = sum(1 for result in results if result.passed)
        failed_count = len(results) - passed_count
        return RetrievalEvalReport(
            retrieval_mode=retrieval_mode,
            retrieval_method=retrieval_method,
            limit=limit,
            case_count=len(results),
            passed_count=passed_count,
            failed_count=failed_count,
            metrics=_compute_metrics(results, limit),
            cases_path=str(Path(cases_path)),
            workspace=str(Path(workspace_path)),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            index_path=str(Path(index_path)) if index_path is not None else None,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            results=results,
        )

    def _case_from_payload(
        self,
        payload: dict[str, Any],
        line_number: int,
    ) -> RetrievalEvalCase:
        known_fields = {
            "id",
            "query",
            "expected_chunk_ids",
            "expected_source_paths",
            "notes",
        }
        case_payload = {
            key: value
            for key, value in payload.items()
            if key in known_fields
        }
        case_payload["metadata"] = {
            key: value
            for key, value in payload.items()
            if key not in known_fields
        }
        try:
            return RetrievalEvalCase.model_validate(case_payload)
        except ValidationError as exc:
            errors = "; ".join(
                _format_validation_error(error)
                for error in exc.errors()
            )
            raise ValueError(
                f"Invalid eval case at line {line_number}: {errors}"
            ) from exc

    def _evaluate_case(
        self,
        *,
        case: RetrievalEvalCase,
        search_results: list[SearchResult],
    ) -> RetrievalEvalCaseResult:
        expected_chunk_ids = set(case.expected_chunk_ids)
        expected_source_paths = tuple(case.expected_source_paths)
        chunk_rank = _first_matching_rank(
            search_results,
            lambda result: result.chunk_id in expected_chunk_ids,
        )
        source_rank = None
        if chunk_rank is None:
            source_rank = _first_matching_rank(
                search_results,
                lambda result: _source_path_matches(
                    result.source_path,
                    expected_source_paths,
                ),
            )

        rank = chunk_rank if chunk_rank is not None else source_rank
        matched_by: Literal["chunk_id", "source_path", "none"]
        if chunk_rank is not None:
            matched_by = "chunk_id"
        elif source_rank is not None:
            matched_by = "source_path"
        else:
            matched_by = "none"

        return RetrievalEvalCaseResult(
            id=case.id,
            query=case.query,
            passed=rank is not None,
            rank=rank,
            reciprocal_rank=(1 / rank) if rank is not None else 0.0,
            matched_by=matched_by,
            expected_chunk_ids=case.expected_chunk_ids,
            expected_source_paths=case.expected_source_paths,
            actual_chunk_ids=[result.chunk_id for result in search_results],
            actual_source_paths=[result.source_path for result in search_results],
            top_results=_compact_top_results(search_results),
        )


def _first_matching_rank(
    search_results: list[SearchResult],
    predicate: Callable[[SearchResult], bool],
) -> int | None:
    for rank, result in enumerate(search_results, start=1):
        if predicate(result):
            return rank
    return None


def _source_path_matches(actual_path: str, expected_paths: tuple[str, ...]) -> bool:
    actual = _normalize_eval_path(actual_path)
    return any(
        actual == expected or actual.endswith(f"/{expected}")
        for expected in (_normalize_eval_path(path) for path in expected_paths)
    )


def _normalize_eval_path(path: str) -> str:
    return path.strip().replace("\\", "/").rstrip("/")


def _compact_top_results(search_results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": result.chunk_id,
            "source_path": result.source_path,
            "score": result.score,
        }
        for rank, result in enumerate(search_results, start=1)
    ]


def _compute_metrics(
    results: list[RetrievalEvalCaseResult],
    limit: int,
) -> dict[str, float]:
    case_count = len(results)
    if case_count == 0:
        raise ValueError("no eval cases found")

    def hit_at(k: int) -> float:
        hits = sum(
            1
            for result in results
            if result.rank is not None and result.rank <= k
        )
        return _round_metric(hits / case_count)

    mrr = sum(result.reciprocal_rank for result in results) / case_count
    return {
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@k": hit_at(limit),
        "mrr": _round_metric(mrr),
    }


def _round_metric(value: float) -> float:
    return round(value, 4)


def _format_validation_error(error: dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "invalid value"))
    if location:
        return f"{location}: {message}"
    return message
