"""RecEvoBench public API."""

from .contracts import EvaluationBundle, RunTrace, TaskManifest
from .scoring import Scorecard, score_bundle
from .sealing import seal_evaluation, validate_public_run

__all__ = [
    "EvaluationBundle",
    "RunTrace",
    "Scorecard",
    "TaskManifest",
    "score_bundle",
    "seal_evaluation",
    "validate_public_run",
]

__version__ = "0.1.0"
