from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from ragent_forge.app.services.evidence_span_service import EvidenceSpan

QuestionType = Literal["factual", "reasoning", "comparison", "how_to"]
Difficulty = Literal["easy", "medium", "hard"]

DEFAULT_ALLOWED_QUESTION_TYPES: tuple[QuestionType, ...] = (
    "factual",
    "reasoning",
    "comparison",
    "how_to",
)
DEFAULT_ALLOWED_DIFFICULTIES: tuple[Difficulty, ...] = (
    "easy",
    "medium",
    "hard",
)
SYSTEM_PROMPT = "You are generating retrieval evaluation cases for a RAG system."
DEFAULT_GENERATION_METHOD = "llm_synthetic_span_v0.2"


class TextGenerationClient(Protocol):
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        ...


@dataclass(frozen=True)
class GeneratedEvalItem:
    query: str
    reference_answer: str
    question_type: QuestionType
    difficulty: Difficulty


@dataclass(frozen=True)
class GeneratedEvalCase:
    id: str
    query: str
    evidence_spans: list[EvidenceSpan]
    reference_answer: str
    question_type: QuestionType
    difficulty: Difficulty
    generation_method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "query": self.query,
            "evidence_spans": [
                _evidence_span_to_record(span) for span in self.evidence_spans
            ],
            "reference_answer": self.reference_answer,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "generation_method": self.generation_method,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record


@dataclass(frozen=True)
class EvalDatasetGenerationReport:
    cases: list[GeneratedEvalCase]
    generated_count: int
    skipped_count: int
    errors: list[dict[str, Any]]
    metadata: dict[str, Any]


class EvalDatasetGenerationService:
    def __init__(
        self,
        generator: TextGenerationClient,
        questions_per_span: int = 2,
        allowed_question_types: tuple[QuestionType, ...] = (
            DEFAULT_ALLOWED_QUESTION_TYPES
        ),
        allowed_difficulties: tuple[Difficulty, ...] = DEFAULT_ALLOWED_DIFFICULTIES,
        generation_method: str = DEFAULT_GENERATION_METHOD,
    ) -> None:
        if questions_per_span < 1:
            raise ValueError("questions_per_span must be greater than 0")
        if not allowed_question_types:
            raise ValueError("allowed_question_types must not be empty")
        if not allowed_difficulties:
            raise ValueError("allowed_difficulties must not be empty")
        if not generation_method.strip():
            raise ValueError("generation_method must be a non-empty string")

        self.generator = generator
        self.questions_per_span = questions_per_span
        self.allowed_question_types = allowed_question_types
        self.allowed_difficulties = allowed_difficulties
        self.generation_method = generation_method.strip()

    def generate(
        self,
        evidence_spans: list[EvidenceSpan],
        max_cases: int | None = None,
    ) -> EvalDatasetGenerationReport:
        if max_cases is not None and max_cases < 0:
            raise ValueError("max_cases must be greater than or equal to 0")

        cases: list[GeneratedEvalCase] = []
        errors: list[dict[str, Any]] = []
        skipped_count = 0

        if max_cases == 0:
            return self._report(cases, skipped_count, errors, len(evidence_spans))

        for span in evidence_spans:
            if max_cases is not None and len(cases) >= max_cases:
                break

            prompt = self._build_prompt(span)
            try:
                response = self.generator.generate_text(
                    prompt,
                    system_prompt=SYSTEM_PROMPT,
                )
                generated_items = self._parse_generated_items(response)
            except Exception as exc:
                errors.append({"span_id": span.id, "message": str(exc)})
                skipped_count += 1
                continue

            for item in generated_items:
                if max_cases is not None and len(cases) >= max_cases:
                    break
                cases.append(
                    GeneratedEvalCase(
                        id=f"synthetic-span-{len(cases) + 1:06d}",
                        query=item.query,
                        evidence_spans=[span],
                        reference_answer=item.reference_answer,
                        question_type=item.question_type,
                        difficulty=item.difficulty,
                        generation_method=self.generation_method,
                    )
                )

        return self._report(cases, skipped_count, errors, len(evidence_spans))

    def _build_prompt(self, span: EvidenceSpan) -> str:
        heading_path = " > ".join(span.heading_path) if span.heading_path else "<none>"
        section_title = span.section_title if span.section_title else "<none>"
        block_types = ", ".join(span.block_types) if span.block_types else "<none>"
        question_types = ", ".join(self.allowed_question_types)
        difficulties = ", ".join(self.allowed_difficulties)
        metadata_lines = [
            f"- source_path: {span.source_path}",
            f"- media_type: {span.media_type}",
            f"- section_title: {section_title}",
            f"- heading_path: {heading_path}",
            f"- block_types: {block_types}",
        ]
        if span.page_start is not None:
            metadata_lines.append(f"- page_start: {span.page_start}")
        if span.page_end is not None:
            metadata_lines.append(f"- page_end: {span.page_end}")

        return "\n".join(
            [
                f"Generate exactly {self.questions_per_span} items.",
                "Each item must be answerable using only the provided evidence.",
                "Do not copy long phrases directly from the evidence.",
                "Questions should sound like realistic user queries.",
                "reference_answer must be grounded only in the evidence.",
                f"Use question_type from: {question_types}.",
                f"Use difficulty from: {difficulties}.",
                "Return strict JSON only.",
                "",
                "Return format:",
                "{",
                '  "items": [',
                "    {",
                '      "query": "...",',
                '      "reference_answer": "...",',
                '      "question_type": "factual|reasoning|comparison|how_to",',
                '      "difficulty": "easy|medium|hard"',
                "    }",
                "  ]",
                "}",
                "",
                "Evidence metadata:",
                *metadata_lines,
                "",
                "Evidence text:",
                span.text,
            ]
        )

    def _parse_generated_items(self, response: str) -> list[GeneratedEvalItem]:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON response: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise ValueError("response must be a JSON object with an items list")

        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        if not items:
            raise ValueError("items must contain at least one item")

        return [
            self._parse_generated_item(item, index)
            for index, item in enumerate(items, start=1)
        ]

    def _parse_generated_item(
        self,
        item: object,
        index: int,
    ) -> GeneratedEvalItem:
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] must be an object")

        query = _required_non_empty_string(item.get("query"), f"items[{index}].query")
        reference_answer = _required_non_empty_string(
            item.get("reference_answer"),
            f"items[{index}].reference_answer",
        )
        question_type = self._parse_question_type(
            item.get("question_type"),
            f"items[{index}].question_type",
        )
        difficulty = self._parse_difficulty(
            item.get("difficulty"),
            f"items[{index}].difficulty",
        )

        return GeneratedEvalItem(
            query=query,
            reference_answer=reference_answer,
            question_type=question_type,
            difficulty=difficulty,
        )

    def _parse_question_type(self, value: object, field_name: str) -> QuestionType:
        question_type = _required_non_empty_string(value, field_name)
        if question_type not in self.allowed_question_types:
            allowed = ", ".join(self.allowed_question_types)
            raise ValueError(f"{field_name} must be one of: {allowed}")
        return cast(QuestionType, question_type)

    def _parse_difficulty(self, value: object, field_name: str) -> Difficulty:
        difficulty = _required_non_empty_string(value, field_name)
        if difficulty not in self.allowed_difficulties:
            allowed = ", ".join(self.allowed_difficulties)
            raise ValueError(f"{field_name} must be one of: {allowed}")
        return cast(Difficulty, difficulty)

    def _report(
        self,
        cases: list[GeneratedEvalCase],
        skipped_count: int,
        errors: list[dict[str, Any]],
        span_count: int,
    ) -> EvalDatasetGenerationReport:
        return EvalDatasetGenerationReport(
            cases=cases,
            generated_count=len(cases),
            skipped_count=skipped_count,
            errors=errors,
            metadata={
                "span_count": span_count,
                "questions_per_span": self.questions_per_span,
                "allowed_question_types": list(self.allowed_question_types),
                "allowed_difficulties": list(self.allowed_difficulties),
                "generation_method": self.generation_method,
            },
        )


def write_jsonl(
    cases: list[GeneratedEvalCase],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output JSONL already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(case.to_jsonl_record(), ensure_ascii=False, sort_keys=True)
        for case in cases
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _required_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _evidence_span_to_record(span: EvidenceSpan) -> dict[str, Any]:
    return {
        "id": span.id,
        "source_path": span.source_path,
        "document_id": span.document_id,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "text": span.text,
        "media_type": span.media_type,
        "section_title": span.section_title,
        "heading_path": list(span.heading_path),
        "block_types": list(span.block_types),
        "page_start": span.page_start,
        "page_end": span.page_end,
        "metadata": span.metadata,
    }
