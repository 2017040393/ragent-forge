from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CompareRunStatus = Literal["success", "failed"]


class RetrievalCompareRun(BaseModel):
    retrieval_mode: str
    retrieval_method: str | None = None
    limit: int
    status: CompareRunStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    passed_count: int | None = None
    failed_count: int | None = None
    case_count: int | None = None
    report_path: str | None = None
    run_dir: str | None = None
    error: str | None = None
    failure_breakdown: dict[str, int] = Field(default_factory=dict)


class RetrievalCompareReport(BaseModel):
    evaluation_type: Literal["retrieval_compare"] = "retrieval_compare"
    cases_path: str
    workspace: str
    retrieval_modes: list[str]
    limits: list[int]
    run_count: int
    success_count: int
    failed_count: int
    runs: list[RetrievalCompareRun]
