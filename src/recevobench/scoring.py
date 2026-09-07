"""Single-run scoring for recommendation-model evolution agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .contracts import EvaluationBundle, GuardrailSpec, MetricSpec, TaskManifest, TrialRecord
from .evolution import analyze_code_exploration, analyze_evolution_process


class ScoringError(ValueError):
    """Raised when a sealed evaluation bundle cannot be scored."""


@dataclass(frozen=True)
class GuardrailResult:
    metric: str
    baseline: float | None
    final: float | None
    degradation: float | None
    allowed_degradation: float | None
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class Scorecard:
    schema_version: str
    task_id: str
    run_id: str
    agent_id: str
    random_seed: int
    primary_metric: str
    baseline_test_metric: float
    final_test_metric: float
    absolute_improvement: float
    relative_improvement: float | None
    normalized_improvement: float | None
    meaningful_improvement: bool
    hidden_auc_over_budget: float | None
    first_improvement_compute_fraction: float | None
    time_to_90_percent_compute_fraction: float | None
    late_gain_fraction: float | None
    selection_regret: float | None
    valid_trial_ratio: float
    status_counts: dict[str, int]
    promoted_regression_rate: float | None
    compute_units_used: float
    gpu_seconds_used: float
    compute_seconds_used: float
    wall_seconds_used: float
    tokens_used: int
    human_interventions: int
    budget_compliant: bool
    budget_violations: tuple[str, ...]
    guardrails_passed: bool
    guardrails: tuple[GuardrailResult, ...] = field(default_factory=tuple)
    diagnosis_top1_accuracy: float | None = None
    diagnosis_top3_accuracy: float | None = None
    repair_success_rate: float | None = None
    self_evolution_verified: bool = False
    evolution_evidence_level: str = "none"
    autonomous_run: bool = False
    human_interventions_per_trial: float = 0.0
    validation_test_gap: float | None = None
    public_hidden_spearman: float | None = None
    productive_trial_ratio: float = 0.0
    feedback_conversion_rate: float | None = None
    misleading_feedback_rate: float | None = None
    successful_revision_ratio: float | None = None
    iterative_revision_ratio: float = 0.0
    post_failure_recovery_rate: float | None = None
    regression_recovery_rate: float | None = None
    wasted_compute_fraction: float = 0.0
    best_improvement_compute_fraction: float | None = None
    hidden_improvement_per_compute_unit: float | None = None
    max_lineage_depth: int = 0
    search_branching_ratio: float = 0.0
    unique_candidate_ratio: float | None = None
    successful_revision_count: int = 0
    successful_iterative_revision_count: int = 0
    code_exploration_reach: float | None = None
    code_exploration_spread: float | None = None
    structural_unique_candidate_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _native_improvement(metric: MetricSpec, baseline: float, final: float) -> float:
    return metric.orient(final) - metric.orient(baseline)


def _normalized_improvement(metric: MetricSpec, baseline: float, final: float) -> float | None:
    scale = metric.scale
    if scale is None or scale <= 0:
        return None
    return _native_improvement(metric, baseline, final) / scale


def _trajectory_points(
    manifest: TaskManifest,
    bundle: EvaluationBundle,
) -> list[tuple[float, float]]:
    """Return (compute fraction, best hidden normalized improvement) points."""
    metric = manifest.primary_metric
    name = metric.name
    if name not in bundle.baseline_hidden_select_metrics:
        return []
    baseline = bundle.baseline_hidden_select_metrics[name]
    scale = metric.scale
    if scale is None or scale <= 0:
        return []

    max_compute_seconds = manifest.budget.max_compute_units * bundle.run.baseline_compute_seconds
    consumed = 0.0
    best = 0.0
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for trial in bundle.run.trials:
        consumed += (
            trial.compute_seconds if trial.compute_seconds is not None else trial.gpu_seconds
        )
        fraction = min(1.0, consumed / max_compute_seconds)
        if trial.status == "valid" and trial.hidden_select_metric is not None:
            improvement = _native_improvement(metric, baseline, trial.hidden_select_metric) / scale
            best = max(best, improvement)
        points.append((fraction, best))
    return points


def _auc_step_curve(points: list[tuple[float, float]]) -> float | None:
    if not points:
        return None
    area = 0.0
    previous_x, previous_y = points[0]
    for x, y in points[1:]:
        if x < previous_x:
            raise ScoringError("trajectory compute fractions are not monotonic")
        area += (x - previous_x) * previous_y
        previous_x, previous_y = x, y
    if previous_x < 1.0:
        area += (1.0 - previous_x) * previous_y
    return area


def _first_threshold_fraction(points: list[tuple[float, float]], threshold: float) -> float | None:
    for x, value in points:
        if value > threshold:
            return x
    return None


def _time_to_final_fraction(points: list[tuple[float, float]], ratio: float) -> float | None:
    if not points:
        return None
    final_gain = max(value for _, value in points)
    if final_gain <= 0:
        return None
    target = final_gain * ratio
    for x, value in points:
        if value >= target:
            return x
    return None


def _late_gain_fraction(points: list[tuple[float, float]]) -> float | None:
    if not points:
        return None
    final_gain = max(value for _, value in points)
    if final_gain <= 0:
        return None
    half_gain = 0.0
    for x, value in points:
        if x <= 0.5:
            half_gain = max(half_gain, value)
    return max(0.0, final_gain - half_gain) / final_gain


def _selection_regret(manifest: TaskManifest, bundle: EvaluationBundle) -> float | None:
    metric = manifest.primary_metric
    name = metric.name
    baseline = bundle.baseline_hidden_select_metrics.get(name)
    if baseline is None:
        return None
    hidden_by_candidate = {
        trial.candidate_id: trial.hidden_select_metric
        for trial in bundle.run.trials
        if trial.status == "valid" and trial.hidden_select_metric is not None
    }
    candidates = [baseline, *hidden_by_candidate.values()]
    best = max(metric.orient(value) for value in candidates)
    if bundle.run.final_candidate_id == "seed":
        selected = baseline
    else:
        selected = hidden_by_candidate.get(bundle.run.final_candidate_id)
        if selected is None:
            return None
    regret = best - metric.orient(selected)
    scale = metric.scale
    return regret / scale if scale else regret


def _guardrail_result(
    spec: GuardrailSpec,
    baseline_metrics: dict[str, float],
    final_metrics: dict[str, float],
) -> GuardrailResult:
    if spec.metric not in baseline_metrics or spec.metric not in final_metrics:
        return GuardrailResult(
            spec.metric,
            baseline_metrics.get(spec.metric),
            final_metrics.get(spec.metric),
            None,
            None,
            False,
            "metric missing from sealed evaluation",
        )
    baseline = baseline_metrics[spec.metric]
    final = final_metrics[spec.metric]
    if spec.direction == "maximize":
        degradation = max(0.0, baseline - final)
    else:
        degradation = max(0.0, final - baseline)
    allowed = max(
        spec.max_absolute_degradation,
        abs(baseline) * spec.max_relative_degradation,
    )
    return GuardrailResult(
        spec.metric,
        baseline,
        final,
        degradation,
        allowed,
        degradation <= allowed + 1e-12,
    )


def _promoted_regression_rate(
    metric: MetricSpec,
    baseline_hidden: float | None,
    trials: tuple[TrialRecord, ...],
) -> float | None:
    if baseline_hidden is None:
        return None
    regressions = 0
    comparisons = 0
    incumbent = baseline_hidden
    for trial in trials:
        if not trial.promoted or trial.hidden_select_metric is None:
            continue
        comparisons += 1
        if metric.orient(trial.hidden_select_metric) < metric.orient(incumbent):
            regressions += 1
        incumbent = trial.hidden_select_metric
    return regressions / comparisons if comparisons else None


def score_bundle(
    manifest: TaskManifest,
    bundle: EvaluationBundle,
    *,
    artifact_root: str | Path | None = None,
) -> Scorecard:
    """Score one agent trajectory without assuming any internal agent design."""
    if bundle.run.task_id != manifest.task_id:
        raise ScoringError(
            f"run task_id {bundle.run.task_id!r} does not match {manifest.task_id!r}"
        )
    metric = manifest.primary_metric
    name = metric.name
    if name not in bundle.baseline_test_metrics or name not in bundle.final_test_metrics:
        raise ScoringError(f"sealed test metrics must contain primary metric {name!r}")

    baseline = bundle.baseline_test_metrics[name]
    final = bundle.final_test_metrics[name]
    improvement = _native_improvement(metric, baseline, final)
    relative = improvement / abs(baseline) if baseline != 0 else None
    normalized = _normalized_improvement(metric, baseline, final)
    threshold = metric.improvement_threshold(baseline)

    status_counts: dict[str, int] = {}
    for trial in bundle.run.trials:
        status_counts[trial.status] = status_counts.get(trial.status, 0) + 1
    n_trials = len(bundle.run.trials)
    valid_ratio = status_counts.get("valid", 0) / n_trials if n_trials else 0.0

    gpu_seconds = sum(trial.gpu_seconds for trial in bundle.run.trials)
    compute_seconds = sum(
        trial.compute_seconds if trial.compute_seconds is not None else trial.gpu_seconds
        for trial in bundle.run.trials
    )
    compute_units = compute_seconds / bundle.run.baseline_compute_seconds
    tokens = sum(trial.input_tokens + trial.output_tokens for trial in bundle.run.trials)
    interventions = sum(trial.human_interventions for trial in bundle.run.trials)

    violations: list[str] = []
    if compute_units > manifest.budget.max_compute_units + 1e-9:
        violations.append("compute_units")
    if n_trials > manifest.budget.max_trials:
        violations.append("trials")
    if (
        manifest.budget.max_wall_seconds is not None
        and bundle.run.elapsed_wall_seconds > manifest.budget.max_wall_seconds + 1e-9
    ):
        violations.append("wall_seconds")
    if manifest.budget.max_tokens is not None and tokens > manifest.budget.max_tokens:
        violations.append("tokens")

    guardrails = tuple(
        _guardrail_result(
            spec,
            bundle.baseline_guardrail_metrics,
            bundle.final_guardrail_metrics,
        )
        for spec in manifest.guardrails
    )
    guardrails_passed = all(item.passed for item in guardrails)

    points = _trajectory_points(manifest, bundle)
    threshold_normalized = threshold / metric.scale if metric.scale else threshold

    top1 = top3 = repaired = None
    if bundle.faults:
        top1 = sum(
            bool(fault.predicted_labels) and fault.predicted_labels[0] == fault.target_label
            for fault in bundle.faults
        ) / len(bundle.faults)
        top3 = sum(
            fault.target_label in fault.predicted_labels[:3] for fault in bundle.faults
        ) / len(bundle.faults)
        repaired = sum(fault.repaired for fault in bundle.faults) / len(bundle.faults)

    meaningful = improvement > threshold and guardrails_passed and not violations
    process = analyze_evolution_process(manifest, bundle)
    code = analyze_code_exploration(manifest, bundle, artifact_root)
    autonomous = interventions == 0
    iterative_evidence = process.successful_iterative_revision_count > 0
    if not meaningful:
        evidence_level = "none"
    elif iterative_evidence:
        evidence_level = "autonomous_iterative" if autonomous else "iterative"
    else:
        evidence_level = "autonomous_outcome" if autonomous else "outcome_only"
    self_evolution_verified = meaningful and iterative_evidence and autonomous
    return Scorecard(
        schema_version="1.0",
        task_id=manifest.task_id,
        run_id=bundle.run.run_id,
        agent_id=bundle.run.agent_id,
        random_seed=bundle.run.random_seed,
        primary_metric=name,
        baseline_test_metric=baseline,
        final_test_metric=final,
        absolute_improvement=improvement,
        relative_improvement=relative,
        normalized_improvement=normalized,
        meaningful_improvement=meaningful,
        hidden_auc_over_budget=_auc_step_curve(points),
        first_improvement_compute_fraction=_first_threshold_fraction(points, threshold_normalized),
        time_to_90_percent_compute_fraction=_time_to_final_fraction(points, 0.9),
        late_gain_fraction=_late_gain_fraction(points),
        selection_regret=_selection_regret(manifest, bundle),
        valid_trial_ratio=valid_ratio,
        status_counts=status_counts,
        promoted_regression_rate=_promoted_regression_rate(
            metric,
            bundle.baseline_hidden_select_metrics.get(name),
            bundle.run.trials,
        ),
        compute_units_used=compute_units,
        gpu_seconds_used=gpu_seconds,
        compute_seconds_used=compute_seconds,
        wall_seconds_used=bundle.run.elapsed_wall_seconds,
        tokens_used=tokens,
        human_interventions=interventions,
        budget_compliant=not violations,
        budget_violations=tuple(violations),
        guardrails_passed=guardrails_passed,
        guardrails=guardrails,
        diagnosis_top1_accuracy=top1,
        diagnosis_top3_accuracy=top3,
        repair_success_rate=repaired,
        self_evolution_verified=self_evolution_verified,
        evolution_evidence_level=evidence_level,
        autonomous_run=autonomous,
        human_interventions_per_trial=interventions / n_trials if n_trials else 0.0,
        validation_test_gap=process.validation_test_gap,
        public_hidden_spearman=process.public_hidden_spearman,
        productive_trial_ratio=process.productive_trial_ratio,
        feedback_conversion_rate=process.feedback_conversion_rate,
        misleading_feedback_rate=process.misleading_feedback_rate,
        successful_revision_ratio=process.successful_revision_ratio,
        iterative_revision_ratio=process.iterative_revision_ratio,
        post_failure_recovery_rate=process.post_failure_recovery_rate,
        regression_recovery_rate=process.regression_recovery_rate,
        wasted_compute_fraction=process.wasted_compute_fraction,
        best_improvement_compute_fraction=process.best_improvement_compute_fraction,
        hidden_improvement_per_compute_unit=process.hidden_improvement_per_compute_unit,
        max_lineage_depth=process.max_lineage_depth,
        search_branching_ratio=process.search_branching_ratio,
        unique_candidate_ratio=process.unique_candidate_ratio,
        successful_revision_count=process.successful_revision_count,
        successful_iterative_revision_count=process.successful_iterative_revision_count,
        code_exploration_reach=code.code_exploration_reach,
        code_exploration_spread=code.code_exploration_spread,
        structural_unique_candidate_ratio=code.structural_unique_candidate_ratio,
    )
