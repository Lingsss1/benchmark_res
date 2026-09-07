"""Canonical recommendation metrics and task evaluators."""

from .evaluator import EvaluationError, evaluate_prepared_split
from .metrics import (
    binary_auc,
    continuous_xauc,
    grouped_auc,
    ranking_metrics,
)

__all__ = [
    "EvaluationError",
    "binary_auc",
    "continuous_xauc",
    "evaluate_prepared_split",
    "grouped_auc",
    "ranking_metrics",
]
