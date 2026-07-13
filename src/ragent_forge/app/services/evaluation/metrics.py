from __future__ import annotations

import math

from ragent_forge.app.services.evaluation.contracts import (
    RetrievalEvalCaseResult,
    RetrievalStageLatencySummary,
)


def precision_at(relevant_result_ranks: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be greater than 0")
    relevant_count = sum(1 for rank in relevant_result_ranks if rank <= k)
    return relevant_count / k


def ndcg_at(
    relevant_result_ranks: list[int],
    *,
    expected_relevant_count: int,
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be greater than 0")
    ideal_relevant_count = min(expected_relevant_count, k)
    if ideal_relevant_count == 0:
        return 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_result_ranks if rank <= k)
    ideal_dcg = sum(
        1 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1)
    )
    return dcg / ideal_dcg


def compute_metrics(
    results: list[RetrievalEvalCaseResult],
    limit: int,
) -> dict[str, float]:
    case_count = len(results)
    if case_count == 0:
        raise ValueError("no eval cases found")

    def hit_at(k: int) -> float:
        hits = sum(
            1 for result in results if result.rank is not None and result.rank <= k
        )
        return round_metric(hits / case_count)

    def result_precision_at(k: int) -> float:
        precision = (
            sum(precision_at(result.relevant_result_ranks, k) for result in results)
            / case_count
        )
        return round_metric(precision)

    mrr = sum(result.reciprocal_rank for result in results) / case_count
    recall = sum(result.recall for result in results) / case_count
    ndcg = sum(result.ndcg for result in results) / case_count
    evidence_coverage, evidence_coverage_case_rate = optional_metric_average(
        [result.evidence_coverage for result in results]
    )
    mapping_coverage, mapping_coverage_case_rate = optional_metric_average(
        [result.mapping_coverage for result in results]
    )
    context_evidence_density = (
        sum(result.context_evidence_density for result in results) / case_count
    )
    duplicate_context_ratio = (
        sum(result.duplicate_context_ratio for result in results) / case_count
    )
    retrieval_latencies = [result.retrieval_latency_ms for result in results]
    avg_retrieval_latency_ms = sum(retrieval_latencies) / case_count
    avg_retrieved_count = sum(result.retrieved_count for result in results) / case_count
    avg_retrieved_context_chars = (
        sum(result.retrieved_context_chars for result in results) / case_count
    )
    avg_estimated_context_tokens = (
        sum(result.estimated_context_tokens for result in results) / case_count
    )
    return {
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@k": hit_at(limit),
        "precision@1": result_precision_at(1),
        "precision@3": result_precision_at(3),
        "precision@5": result_precision_at(5),
        "precision@k": result_precision_at(limit),
        "mrr": round_metric(mrr),
        "recall@k": round_metric(recall),
        "ndcg@k": round_metric(ndcg),
        "evidence_coverage@k": round_metric(evidence_coverage),
        "evidence_coverage_case_rate": round_metric(evidence_coverage_case_rate),
        "mapping_coverage": round_metric(mapping_coverage),
        "mapping_coverage_case_rate": round_metric(mapping_coverage_case_rate),
        "context_evidence_density": round_metric(context_evidence_density),
        "duplicate_context_ratio": round_metric(duplicate_context_ratio),
        "avg_retrieval_latency_ms": round_metric(avg_retrieval_latency_ms),
        "retrieval_latency_p50_ms": round_metric(percentile(retrieval_latencies, 0.5)),
        "retrieval_latency_p95_ms": round_metric(percentile(retrieval_latencies, 0.95)),
        "avg_retrieved_count": round_metric(avg_retrieved_count),
        "avg_retrieved_context_chars": round_metric(avg_retrieved_context_chars),
        "avg_estimated_context_tokens": round_metric(avg_estimated_context_tokens),
    }


def summarize_stage_latencies(
    results: list[RetrievalEvalCaseResult],
) -> dict[str, RetrievalStageLatencySummary]:
    samples: dict[str, list[float]] = {}
    for result in results:
        pipeline = result.metadata.get("retrieval_pipeline")
        if not isinstance(pipeline, list):
            continue
        for stage in pipeline:
            if not isinstance(stage, dict):
                continue
            name = stage.get("name")
            latency = stage.get("latency_ms")
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(latency, int | float) or isinstance(latency, bool):
                continue
            samples.setdefault(name, []).append(float(latency))

    return {
        name: RetrievalStageLatencySummary(
            sample_count=len(values),
            average_ms=round_metric(sum(values) / len(values)),
            p50_ms=round_metric(percentile(values, 0.5)),
            p95_ms=round_metric(percentile(values, 0.95)),
        )
        for name, values in sorted(samples.items())
    }


def optional_metric_average(values: list[float | None]) -> tuple[float, float]:
    available = [value for value in values if value is not None]
    if not available:
        return 0.0, 0.0
    return sum(available) / len(available), len(available) / len(values)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if percentile_value < 0 or percentile_value > 1:
        raise ValueError("percentile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def round_metric(value: float) -> float:
    return round(value, 4)
