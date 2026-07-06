import json
from pathlib import Path

import pytest

from ragent_forge.app.services.eval_dataset_generation_service import (
    EvalDatasetGenerationService,
    write_jsonl,
)
from ragent_forge.app.services.evidence_span_service import EvidenceSpan
from ragent_forge.app.services.retrieval_eval_service import RetrievalEvalService


class FakeGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str | None]] = []

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        if not self.responses:
            raise AssertionError("unexpected generator call")
        return self.responses.pop(0)


def test_generates_cases_from_one_span_with_fake_generator() -> None:
    generator = FakeGenerator(
        [
            _items_response(
                [
                    {
                        "query": "How does hybrid retrieval keep evidence useful?",
                        "reference_answer": (
                            "It combines lexical and semantic retrieval while "
                            "keeping evidence inspectable."
                        ),
                        "question_type": "reasoning",
                        "difficulty": "medium",
                    },
                    {
                        "query": "What retrieval modes are combined?",
                        "reference_answer": (
                            "The evidence says lexical and semantic retrieval "
                            "are combined."
                        ),
                        "question_type": "factual",
                        "difficulty": "easy",
                    },
                ]
            )
        ]
    )
    span = make_span()

    report = EvalDatasetGenerationService(generator=generator).generate([span])

    assert report.generated_count == 2
    assert report.skipped_count == 0
    assert report.errors == []
    assert [case.id for case in report.cases] == [
        "synthetic-span-000001",
        "synthetic-span-000002",
    ]
    first_case = report.cases[0]
    assert first_case.query == "How does hybrid retrieval keep evidence useful?"
    assert first_case.reference_answer.startswith("It combines")
    assert first_case.question_type == "reasoning"
    assert first_case.difficulty == "medium"
    assert first_case.evidence_spans == [span]
    assert first_case.generation_method == "llm_synthetic_span_v0.2"
    assert generator.calls[0][1] == (
        "You are generating retrieval evaluation cases for a RAG system."
    )


def test_generated_jsonl_record_is_span_based_without_chunk_or_source_expectations(
    tmp_path: Path,
) -> None:
    generator = FakeGenerator(
        [
            _items_response(
                [
                    {
                        "query": "Why use span-based eval cases?",
                        "reference_answer": (
                            "They stay grounded in stable evidence spans rather "
                            "than workspace chunk ids."
                        ),
                        "question_type": "reasoning",
                        "difficulty": "medium",
                    }
                ]
            )
        ]
    )

    report = EvalDatasetGenerationService(
        generator=generator,
        questions_per_span=1,
    ).generate([make_span()])
    record = report.cases[0].to_jsonl_record()

    assert record["evidence_spans"][0]["id"] == "docs/rag.md::span-0001"
    assert record["reference_answer"].startswith("They stay grounded")
    assert record["question_type"] == "reasoning"
    assert record["difficulty"] == "medium"
    assert record["generation_method"] == "llm_synthetic_span_v0.2"
    assert "expected_chunk_ids" not in record
    assert "expected_source_paths" not in record

    output_path = write_jsonl(report.cases, tmp_path / "cases.jsonl")
    loaded_cases = RetrievalEvalService().load_cases(output_path)

    assert len(loaded_cases) == 1
    assert loaded_cases[0].expected_chunk_ids == []
    assert loaded_cases[0].expected_source_paths == []
    assert loaded_cases[0].evidence_spans[0].id == "docs/rag.md::span-0001"
    assert loaded_cases[0].metadata["reference_answer"].startswith(
        "They stay grounded"
    )
    assert loaded_cases[0].metadata["generation_method"] == (
        "llm_synthetic_span_v0.2"
    )


def test_multiple_spans_produce_deterministic_ids_and_max_cases_limits_output() -> None:
    first_response = _items_response(
        [
            {
                "query": "First query?",
                "reference_answer": "First answer.",
                "question_type": "factual",
                "difficulty": "easy",
            },
            {
                "query": "Second query?",
                "reference_answer": "Second answer.",
                "question_type": "reasoning",
                "difficulty": "medium",
            },
        ]
    )
    second_response = _items_response(
        [
            {
                "query": "Third query?",
                "reference_answer": "Third answer.",
                "question_type": "comparison",
                "difficulty": "hard",
            },
            {
                "query": "Fourth query?",
                "reference_answer": "Fourth answer.",
                "question_type": "how_to",
                "difficulty": "medium",
            },
        ]
    )
    generator = FakeGenerator([first_response, second_response])
    spans = [
        make_span(span_id="docs/rag.md::span-0001"),
        make_span(span_id="docs/rag.md::span-0002", text="Second span evidence."),
    ]

    report = EvalDatasetGenerationService(generator=generator).generate(
        spans,
        max_cases=3,
    )

    assert [case.id for case in report.cases] == [
        "synthetic-span-000001",
        "synthetic-span-000002",
        "synthetic-span-000003",
    ]
    assert [case.query for case in report.cases] == [
        "First query?",
        "Second query?",
        "Third query?",
    ]
    assert report.generated_count == 3
    assert len(generator.calls) == 2


def test_invalid_json_response_records_error_and_skips_span() -> None:
    service = EvalDatasetGenerationService(generator=FakeGenerator(["not json"]))

    report = service.generate([make_span()])

    assert report.cases == []
    assert report.generated_count == 0
    assert report.skipped_count == 1
    assert report.errors[0]["span_id"] == "docs/rag.md::span-0001"
    assert "invalid JSON" in report.errors[0]["message"]


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (
            {
                "query": "Question?",
                "reference_answer": "Answer.",
                "question_type": "unsupported",
                "difficulty": "easy",
            },
            "question_type",
        ),
        (
            {
                "query": "Question?",
                "reference_answer": "Answer.",
                "question_type": "factual",
                "difficulty": "impossible",
            },
            "difficulty",
        ),
        (
            {
                "query": "   ",
                "reference_answer": "Answer.",
                "question_type": "factual",
                "difficulty": "easy",
            },
            "query",
        ),
        (
            {
                "query": "Question?",
                "reference_answer": "",
                "question_type": "factual",
                "difficulty": "easy",
            },
            "reference_answer",
        ),
    ],
)
def test_invalid_generated_items_are_rejected(
    item: dict[str, str],
    message: str,
) -> None:
    report = EvalDatasetGenerationService(
        generator=FakeGenerator([_items_response([item])]),
        questions_per_span=1,
    ).generate([make_span()])

    assert report.cases == []
    assert report.skipped_count == 1
    assert message in report.errors[0]["message"]


def test_missing_or_non_list_items_are_rejected() -> None:
    missing_items = EvalDatasetGenerationService(
        generator=FakeGenerator([json.dumps({"cases": []})])
    ).generate([make_span()])
    non_list_items = EvalDatasetGenerationService(
        generator=FakeGenerator([json.dumps({"items": {}})])
    ).generate([make_span()])

    assert "items" in missing_items.errors[0]["message"]
    assert "items" in non_list_items.errors[0]["message"]
    assert missing_items.skipped_count == 1
    assert non_list_items.skipped_count == 1


def test_too_few_items_are_rejected() -> None:
    service = EvalDatasetGenerationService(
        generator=FakeGenerator(
            [
                _items_response(
                    [
                        {
                            "query": "Only one question?",
                            "reference_answer": "Only one answer.",
                            "question_type": "factual",
                            "difficulty": "easy",
                        }
                    ]
                )
            ]
        ),
        questions_per_span=2,
    )

    report = service.generate([make_span()])

    assert report.cases == []
    assert report.skipped_count == 1
    assert report.errors[0]["message"] == "items must contain exactly 2 items"


def test_too_many_items_are_rejected() -> None:
    service = EvalDatasetGenerationService(
        generator=FakeGenerator(
            [
                _items_response(
                    [
                        {
                            "query": "First question?",
                            "reference_answer": "First answer.",
                            "question_type": "factual",
                            "difficulty": "easy",
                        },
                        {
                            "query": "Second question?",
                            "reference_answer": "Second answer.",
                            "question_type": "reasoning",
                            "difficulty": "medium",
                        },
                        {
                            "query": "Third question?",
                            "reference_answer": "Third answer.",
                            "question_type": "comparison",
                            "difficulty": "hard",
                        },
                    ]
                )
            ]
        ),
        questions_per_span=2,
    )

    report = service.generate([make_span()])

    assert report.cases == []
    assert report.skipped_count == 1
    assert report.errors[0]["message"] == "items must contain exactly 2 items"


def test_exact_questions_per_span_item_count_still_works() -> None:
    service = EvalDatasetGenerationService(
        generator=FakeGenerator(
            [
                _items_response(
                    [
                        {
                            "query": "First question?",
                            "reference_answer": "First answer.",
                            "question_type": "factual",
                            "difficulty": "easy",
                        },
                        {
                            "query": "Second question?",
                            "reference_answer": "Second answer.",
                            "question_type": "reasoning",
                            "difficulty": "medium",
                        },
                        {
                            "query": "Third question?",
                            "reference_answer": "Third answer.",
                            "question_type": "comparison",
                            "difficulty": "hard",
                        },
                    ]
                )
            ]
        ),
        questions_per_span=3,
    )

    report = service.generate([make_span()])

    assert report.generated_count == 3
    assert report.skipped_count == 0
    assert report.errors == []
    assert [case.id for case in report.cases] == [
        "synthetic-span-000001",
        "synthetic-span-000002",
        "synthetic-span-000003",
    ]


def test_write_jsonl_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    case = EvalDatasetGenerationService(
        generator=FakeGenerator(
            [
                _items_response(
                    [
                        {
                            "query": "Question?",
                            "reference_answer": "Answer.",
                            "question_type": "factual",
                            "difficulty": "easy",
                        }
                    ]
                )
            ]
        ),
        questions_per_span=1,
    ).generate([make_span()]).cases[0]
    output_path = tmp_path / "cases.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_jsonl([case], output_path)

    write_jsonl([case], output_path, overwrite=True)

    assert json.loads(output_path.read_text(encoding="utf-8"))["id"] == (
        "synthetic-span-000001"
    )


def test_prompt_includes_span_metadata_and_evidence_text() -> None:
    span = make_span(
        source_path="papers/eval.pdf",
        media_type="application/pdf",
        section_title="Evaluation",
        heading_path=(),
        page_start=2,
        page_end=3,
        block_types=("paragraph", "table"),
        text="Evidence text about page-aware PDF spans.",
    )
    generator = FakeGenerator(
        [
            _items_response(
                [
                    {
                        "query": "What does the PDF evidence explain?",
                        "reference_answer": "It explains page-aware PDF spans.",
                        "question_type": "factual",
                        "difficulty": "easy",
                    }
                ]
            )
        ]
    )

    EvalDatasetGenerationService(
        generator=generator,
        questions_per_span=1,
    ).generate([span])

    prompt, _system_prompt = generator.calls[0]
    assert "Generate exactly 1 items" in prompt
    assert "answerable using only the provided evidence" in prompt
    assert "Do not copy long phrases directly" in prompt
    assert "strict JSON only" in prompt
    assert "papers/eval.pdf" in prompt
    assert "application/pdf" in prompt
    assert "Evaluation" in prompt
    assert "page_start: 2" in prompt
    assert "page_end: 3" in prompt
    assert "paragraph, table" in prompt
    assert "Evidence text about page-aware PDF spans." in prompt


def make_span(
    span_id: str = "docs/rag.md::span-0001",
    *,
    source_path: str = "docs/rag.md",
    media_type: str = "text/markdown",
    section_title: str | None = "Hybrid Retrieval",
    heading_path: tuple[str, ...] = ("RAG", "Hybrid Retrieval"),
    page_start: int | None = None,
    page_end: int | None = None,
    block_types: tuple[str, ...] = ("paragraph",),
    text: str = (
        "Hybrid retrieval combines lexical and semantic retrieval while "
        "keeping the supporting evidence inspectable."
    ),
) -> EvidenceSpan:
    return EvidenceSpan(
        id=span_id,
        source_path=source_path,
        document_id=source_path,
        start_char=None if media_type == "application/pdf" else 10,
        end_char=None if media_type == "application/pdf" else 120,
        text=text,
        media_type=media_type,
        section_title=section_title,
        heading_path=heading_path,
        block_types=block_types,
        page_start=page_start,
        page_end=page_end,
        metadata={"text_sha256": "abc123"},
    )


def _items_response(items: list[dict[str, str]]) -> str:
    return json.dumps({"items": items})
