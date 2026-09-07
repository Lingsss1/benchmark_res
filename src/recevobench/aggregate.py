"""Aggregation across repeated runs of the same agent/task configuration."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .scoring import Scorecard


@dataclass(frozen=True)
class AggregateScorecard:
    n_runs: int
    n_tasks: int
    task_ids: tuple[str, ...]
    agent_ids: tuple[str, ...]
    runs_per_task: dict[str, int]
    mean_normalized_improvement: float | None
    std_normalized_improvement: float | None
    probability_of_meaningful_improvement: float
    cvar20_normalized_improvement: float | None
    mean_hidden_auc_over_budget: float | None
    mean_valid_trial_ratio: float
    budget_compliance_rate: float
    guardrail_pass_rate: float
    probability_of_verified_self_evolution: float
    autonomous_run_rate: float
    mean_productive_trial_ratio: float
    mean_feedback_conversion_rate: float | None
    mean_validation_test_gap: float | None
    mean_wasted_compute_fraction: float
    mean_code_exploration_reach: float | None
    mean_structural_unique_candidate_ratio: float | None
    mean_max_lineage_depth: float
    worst_task_mean_normalized_improvement: float | None
    cross_task_std_normalized_improvement: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "n_tasks": self.n_tasks,
            "task_ids": list(self.task_ids),
            "agent_ids": list(self.agent_ids),
            "runs_per_task": self.runs_per_task,
            "mean_normalized_improvement": self.mean_normalized_improvement,
            "std_normalized_improvement": self.std_normalized_improvement,
            "probability_of_meaningful_improvement": self.probability_of_meaningful_improvement,
            "cvar20_normalized_improvement": self.cvar20_normalized_improvement,
            "mean_hidden_auc_over_budget": self.mean_hidden_auc_over_budget,
            "mean_valid_trial_ratio": self.mean_valid_trial_ratio,
            "budget_compliance_rate": self.budget_compliance_rate,
            "guardrail_pass_rate": self.guardrail_pass_rate,
            "probability_of_verified_self_evolution": (self.probability_of_verified_self_evolution),
            "autonomous_run_rate": self.autonomous_run_rate,
            "mean_productive_trial_ratio": self.mean_productive_trial_ratio,
            "mean_feedback_conversion_rate": self.mean_feedback_conversion_rate,
            "mean_validation_test_gap": self.mean_validation_test_gap,
            "mean_wasted_compute_fraction": self.mean_wasted_compute_fraction,
            "mean_code_exploration_reach": self.mean_code_exploration_reach,
            "mean_structural_unique_candidate_ratio": (self.mean_structural_unique_candidate_ratio),
            "mean_max_lineage_depth": self.mean_max_lineage_depth,
            "worst_task_mean_normalized_improvement": (self.worst_task_mean_normalized_improvement),
            "cross_task_std_normalized_improvement": (self.cross_task_std_normalized_improvement),
        }


def _mean_optional(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def aggregate_scorecards(scorecards: Iterable[Scorecard]) -> AggregateScorecard:
    cards = list(scorecards)
    if not cards:
        raise ValueError("at least one scorecard is required")

    by_task: dict[str, list[Scorecard]] = {}
    for card in cards:
        by_task.setdefault(card.task_id, []).append(card)

    task_improvements = {
        task_id: [
            card.normalized_improvement
            for card in task_cards
            if card.normalized_improvement is not None
        ]
        for task_id, task_cards in by_task.items()
    }
    task_means = [fmean(values) for values in task_improvements.values() if values]
    macro_mean = fmean(task_means) if task_means else None
    task_mean_variance = (
        fmean((value - macro_mean) ** 2 for value in task_means)
        if macro_mean is not None and task_means
        else None
    )
    balanced_variance = None
    if macro_mean is not None:
        balanced_variance = fmean(
            fmean((value - macro_mean) ** 2 for value in values)
            for values in task_improvements.values()
            if values
        )

    task_cvars: list[float] = []
    for values in task_improvements.values():
        if not values:
            continue
        tail_size = max(1, math.ceil(len(values) * 0.2))
        task_cvars.append(fmean(sorted(values)[:tail_size]))

    def task_macro(attribute: str) -> float:
        return fmean(
            fmean(float(getattr(card, attribute)) for card in task_cards)
            for task_cards in by_task.values()
        )

    def task_macro_optional(attribute: str) -> float | None:
        per_task = [
            _mean_optional(getattr(card, attribute) for card in task_cards)
            for task_cards in by_task.values()
        ]
        present = [value for value in per_task if value is not None]
        return fmean(present) if present else None

    return AggregateScorecard(
        n_runs=len(cards),
        n_tasks=len(by_task),
        task_ids=tuple(sorted({card.task_id for card in cards})),
        agent_ids=tuple(sorted({card.agent_id for card in cards})),
        runs_per_task={task_id: len(task_cards) for task_id, task_cards in sorted(by_task.items())},
        mean_normalized_improvement=macro_mean,
        std_normalized_improvement=(
            math.sqrt(balanced_variance) if balanced_variance is not None else None
        ),
        probability_of_meaningful_improvement=task_macro("meaningful_improvement"),
        cvar20_normalized_improvement=(fmean(task_cvars) if task_cvars else None),
        mean_hidden_auc_over_budget=task_macro_optional("hidden_auc_over_budget"),
        mean_valid_trial_ratio=task_macro("valid_trial_ratio"),
        budget_compliance_rate=task_macro("budget_compliant"),
        guardrail_pass_rate=task_macro("guardrails_passed"),
        probability_of_verified_self_evolution=task_macro("self_evolution_verified"),
        autonomous_run_rate=task_macro("autonomous_run"),
        mean_productive_trial_ratio=task_macro("productive_trial_ratio"),
        mean_feedback_conversion_rate=task_macro_optional("feedback_conversion_rate"),
        mean_validation_test_gap=task_macro_optional("validation_test_gap"),
        mean_wasted_compute_fraction=task_macro("wasted_compute_fraction"),
        mean_code_exploration_reach=task_macro_optional("code_exploration_reach"),
        mean_structural_unique_candidate_ratio=task_macro_optional(
            "structural_unique_candidate_ratio"
        ),
        mean_max_lineage_depth=task_macro("max_lineage_depth"),
        worst_task_mean_normalized_improvement=(min(task_means) if task_means else None),
        cross_task_std_normalized_improvement=(
            math.sqrt(task_mean_variance) if task_mean_variance is not None else None
        ),
    )
