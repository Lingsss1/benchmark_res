"""Trusted-evaluator boundary for joining public traces with hidden results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .contracts import EvaluationBundle, FaultEvaluation, RunTrace


class SealingError(ValueError):
    """Raised when an agent submission crosses the hidden-evaluation boundary."""


def validate_public_run(public_run: RunTrace) -> None:
    """Reject hidden values and non-final valid candidates at the agent boundary."""

    leaked = [
        trial.candidate_id for trial in public_run.trials if trial.hidden_select_metric is not None
    ]
    if leaked:
        raise SealingError("public trace contains hidden_select_metric for: " + ", ".join(leaked))
    if public_run.final_candidate_id != "seed":
        final = next(
            trial
            for trial in public_run.trials
            if trial.candidate_id == public_run.final_candidate_id
        )
        if final.status != "valid":
            raise SealingError("final_candidate_id must refer to a valid trial or seed")


def seal_evaluation(
    public_run: RunTrace,
    *,
    hidden_select_by_candidate: Mapping[str, float],
    baseline_hidden_select_metrics: Mapping[str, float],
    baseline_test_metrics: Mapping[str, float],
    final_test_metrics: Mapping[str, float],
    baseline_guardrail_metrics: Mapping[str, float] | None = None,
    final_guardrail_metrics: Mapping[str, float] | None = None,
    baseline_public_metrics: Mapping[str, float] | None = None,
    faults: Sequence[FaultEvaluation] = (),
) -> EvaluationBundle:
    """Create the canonical sealed artifact without trusting agent-supplied hidden data.

    A valid public submission contains no hidden-select values.  The trusted
    evaluator must provide one hidden value for every valid candidate so that
    trajectory scoring cannot silently omit an inconvenient experiment.
    """

    validate_public_run(public_run)

    valid_ids = {trial.candidate_id for trial in public_run.trials if trial.status == "valid"}
    supplied_ids = set(hidden_select_by_candidate)
    missing = sorted(valid_ids - supplied_ids)
    unknown = sorted(supplied_ids - valid_ids)
    if missing:
        raise SealingError("missing hidden-select results for: " + ", ".join(missing))
    if unknown:
        raise SealingError(
            "hidden-select results contain unknown candidates: " + ", ".join(unknown)
        )

    sealed_trials = tuple(
        replace(
            trial,
            hidden_select_metric=(
                float(hidden_select_by_candidate[trial.candidate_id])
                if trial.status == "valid"
                else None
            ),
        )
        for trial in public_run.trials
    )
    sealed_run = replace(public_run, trials=sealed_trials)
    return EvaluationBundle(
        run=sealed_run,
        baseline_hidden_select_metrics=dict(baseline_hidden_select_metrics),
        baseline_test_metrics=dict(baseline_test_metrics),
        final_test_metrics=dict(final_test_metrics),
        baseline_guardrail_metrics=dict(baseline_guardrail_metrics or {}),
        final_guardrail_metrics=dict(final_guardrail_metrics or {}),
        faults=tuple(faults),
        baseline_public_metrics=dict(baseline_public_metrics or {}),
    )
