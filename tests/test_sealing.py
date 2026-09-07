from dataclasses import replace
from pathlib import Path

import pytest

from recevobench.io import load_evaluation_bundle
from recevobench.sealing import SealingError, seal_evaluation, validate_public_run

ROOT = Path(__file__).resolve().parents[1]


def _public_run():
    bundle = load_evaluation_bundle(ROOT / "examples" / "amazon-evaluation.example.json")
    return replace(
        bundle.run,
        trials=tuple(replace(trial, hidden_select_metric=None) for trial in bundle.run.trials),
    )


def test_trusted_evaluator_joins_hidden_metrics() -> None:
    sealed = seal_evaluation(
        _public_run(),
        hidden_select_by_candidate={"cand-001": 0.198, "cand-003": 0.216},
        baseline_hidden_select_metrics={"NDCG@10": 0.190},
        baseline_test_metrics={"NDCG@10": 0.188},
        final_test_metrics={"NDCG@10": 0.211},
    )
    assert [trial.hidden_select_metric for trial in sealed.run.trials] == [
        0.198,
        None,
        0.216,
    ]


def test_public_trace_cannot_supply_hidden_results() -> None:
    run = _public_run()
    leaked = replace(
        run,
        trials=(replace(run.trials[0], hidden_select_metric=0.999), *run.trials[1:]),
    )
    with pytest.raises(SealingError, match="public trace contains"):
        seal_evaluation(
            leaked,
            hidden_select_by_candidate={"cand-001": 0.198, "cand-003": 0.216},
            baseline_hidden_select_metrics={"NDCG@10": 0.190},
            baseline_test_metrics={"NDCG@10": 0.188},
            final_test_metrics={"NDCG@10": 0.211},
        )


def test_hidden_evaluator_cannot_omit_a_valid_candidate() -> None:
    with pytest.raises(SealingError, match="missing hidden-select"):
        seal_evaluation(
            _public_run(),
            hidden_select_by_candidate={"cand-001": 0.198},
            baseline_hidden_select_metrics={"NDCG@10": 0.190},
            baseline_test_metrics={"NDCG@10": 0.188},
            final_test_metrics={"NDCG@10": 0.211},
        )


def test_final_candidate_must_be_valid() -> None:
    run = _public_run()
    invalid_final = replace(run, final_candidate_id="cand-002")
    with pytest.raises(SealingError, match="valid trial"):
        validate_public_run(invalid_final)
