from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragent_forge.app.services.evaluation.contracts import RetrievalEvalCase


def load_cases(cases_path: str | Path) -> list[RetrievalEvalCase]:
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
                f"Invalid JSONL in eval cases {path} at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid eval case at line {line_number}: expected object"
            )
        cases.append(_case_from_payload(payload, line_number))

    if not cases:
        raise ValueError("no eval cases found")
    return cases


def _case_from_payload(
    payload: dict[str, Any],
    line_number: int,
) -> RetrievalEvalCase:
    known_fields = {
        "id",
        "query",
        "expected_chunk_ids",
        "expected_source_paths",
        "evidence_spans",
        "notes",
    }
    case_payload = {key: value for key, value in payload.items() if key in known_fields}
    case_payload["metadata"] = {
        key: value for key, value in payload.items() if key not in known_fields
    }
    try:
        return RetrievalEvalCase.model_validate(case_payload)
    except ValidationError as exc:
        errors = "; ".join(_format_validation_error(error) for error in exc.errors())
        raise ValueError(f"Invalid eval case at line {line_number}: {errors}") from exc


def _format_validation_error(error: Mapping[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "invalid value"))
    if location:
        return f"{location}: {message}"
    return message
