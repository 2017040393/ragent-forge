"""Compatibility facade for the split retrieval evaluation service.

New code should import the focused modules under ``services.evaluation``.
This path remains stable for integrations and existing evaluation fixtures.
"""

from ragent_forge.app.services.evaluation.contracts import (
    FailureType,
    MatchedBy,
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
    RetrievalEvalReport,
    RetrievalRunnerProtocol,
    RetrievalStageLatencySummary,
    SearchServiceProtocol,
    WorkspaceChunksProtocol,
)
from ragent_forge.app.services.evaluation.metrics import (
    compute_metrics as _compute_metrics,
)
from ragent_forge.app.services.evaluation.metrics import (
    percentile as _percentile,
)
from ragent_forge.app.services.evaluation.metrics import (
    round_metric as _round_metric,
)
from ragent_forge.app.services.evaluation.reporting import (
    classify_failure as _classify_failure,
)
from ragent_forge.app.services.evaluation.runner import RetrievalEvalService

__all__ = [
    "FailureType",
    "MatchedBy",
    "RetrievalEvalCase",
    "RetrievalEvalCaseResult",
    "RetrievalEvalReport",
    "RetrievalEvalService",
    "RetrievalStageLatencySummary",
    "RetrievalRunnerProtocol",
    "SearchServiceProtocol",
    "WorkspaceChunksProtocol",
    "_classify_failure",
    "_compute_metrics",
    "_percentile",
    "_round_metric",
]
