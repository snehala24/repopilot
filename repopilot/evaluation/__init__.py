"""RepoPilot evaluation package."""

from repopilot.evaluation.dataset import (
    GoldenQuestion,
    QuestionCategory,
    load_golden_dataset,
    save_golden_dataset,
)
from repopilot.evaluation.metrics import (
    RetrievalMetrics,
    SourceType,
    SourceTypeBreakdown,
    classify_source_type,
    compute_retrieval_metrics,
    compute_source_breakdown,
)
from repopilot.evaluation.runner import (
    EvaluationSummary,
    QuestionEvalResult,
    run_retrieval_evaluation,
)

__all__ = [
    "EvaluationSummary",
    "GoldenQuestion",
    "QuestionCategory",
    "QuestionEvalResult",
    "RetrievalMetrics",
    "SourceType",
    "SourceTypeBreakdown",
    "classify_source_type",
    "compute_retrieval_metrics",
    "compute_source_breakdown",
    "load_golden_dataset",
    "run_retrieval_evaluation",
    "save_golden_dataset",
]
