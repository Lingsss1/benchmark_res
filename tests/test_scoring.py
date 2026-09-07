from dataclasses import replace
from pathlib import Path

import pytest

from recevobench.aggregate import aggregate_scorecards
from recevobench.contracts import MetricSpec
from recevobench.io import load_evaluation_bundle, load_manifest
from recevobench.scoring import _native_improvement, score_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def amazon():
    return (
        load_manifest(ROOT / "tasks" / "amazon-seqrec-v0.json"),
        load_evaluation_bundle(ROOT / "examples" / "amazon-evaluation.example.json"),
    )


def test_example_scorecard_covers_outcome_process_and_fault_metrics(amazon) -> None:
    manifest, bundle = amazon
    card = score_bundle(manifest, bundle)

    assert card.absolute_improvement == pytest.approx(0.023)
    assert card.normalized_improvement == pytest.approx(0.023)
    assert card.relative_improvement == pytest.approx(0.023 / 0.188)
    assert card.meaningful_improvement is True
    assert card.hidden_auc_over_budget == pytest.approx(0.01595)
    assert card.first_improvement_compute_fraction == pytest.approx(0.1875)
    assert card.time_to_90_percent_compute_fraction == pytest.approx(0.475)
    assert card.late_gain_fraction == pytest.approx(0.0)
    assert card.selection_regret == pytest.approx(0.0)
    assert card.valid_trial_ratio == pytest.approx(2 / 3)
    assert card.status_counts == {"valid": 2, "oom": 1}
    assert card.compute_units_used == pytest.approx(11.4)
    assert card.tokens_used == 43_500
    assert card.guardrails_passed is True
    assert card.diagnosis_top1_accuracy == pytest.approx(0.5)
    assert card.diagnosis_top3_accuracy == pytest.approx(1.0)
    assert card.repair_success_rate == pytest.approx(0.5)
    assert card.validation_test_gap == pytest.approx(0.003)
    assert card.public_hidden_spearman == pytest.approx(1.0)
    assert card.productive_trial_ratio == pytest.approx(1.0)
    assert card.feedback_conversion_rate == pytest.approx(1.0)
    assert card.misleading_feedback_rate == pytest.approx(0.0)
    assert card.successful_revision_ratio == pytest.approx(1.0)
    assert card.iterative_revision_ratio == pytest.approx(0.5)
    assert card.post_failure_recovery_rate == pytest.approx(1.0)
    assert card.wasted_compute_fraction == pytest.approx(180 / 1140)
    assert card.best_improvement_compute_fraction == pytest.approx(0.475)
    assert card.max_lineage_depth == 2
    assert card.search_branching_ratio == pytest.approx(0.5)
    assert card.unique_candidate_ratio == pytest.approx(1.0)
    assert card.successful_iterative_revision_count == 1
    assert card.self_evolution_verified is True
    assert card.evolution_evidence_level == "autonomous_iterative"


def test_guardrail_failure_prevents_meaningful_improvement(amazon) -> None:
    manifest, bundle = amazon
    degraded = replace(
        bundle,
        final_guardrail_metrics={
            **bundle.final_guardrail_metrics,
            "CatalogCoverage@10": 0.01,
        },
    )
    card = score_bundle(manifest, degraded)
    assert card.absolute_improvement > 0
    assert card.guardrails_passed is False
    assert card.meaningful_improvement is False


def test_one_shot_seed_improvement_is_not_verified_as_iterative_evolution(amazon) -> None:
    manifest, bundle = amazon
    first = bundle.run.trials[0]
    one_shot = replace(
        bundle,
        run=replace(bundle.run, trials=(first,), final_candidate_id=first.candidate_id),
    )
    card = score_bundle(manifest, one_shot)

    assert card.meaningful_improvement is True
    assert card.successful_revision_count == 1
    assert card.successful_iterative_revision_count == 0
    assert card.self_evolution_verified is False
    assert card.evolution_evidence_level == "autonomous_outcome"


def test_successful_side_branch_does_not_verify_final_candidate_lineage(amazon) -> None:
    manifest, bundle = amazon
    changed = replace(bundle, run=replace(bundle.run, final_candidate_id="cand-001"))
    card = score_bundle(manifest, changed)

    assert card.successful_revision_count == 2
    assert card.successful_iterative_revision_count == 0
    assert card.self_evolution_verified is False
    assert card.evolution_evidence_level == "autonomous_outcome"


def test_budget_violation_prevents_meaningful_improvement(amazon) -> None:
    manifest, bundle = amazon
    over_budget = replace(bundle, run=replace(bundle.run, elapsed_wall_seconds=50_000))
    card = score_bundle(manifest, over_budget)
    assert card.budget_compliant is False
    assert card.budget_violations == ("wall_seconds",)
    assert card.meaningful_improvement is False


def test_lower_is_better_metric_orientation() -> None:
    metric = MetricSpec("MAE", "minimize", 0.0, 1.0)
    assert _native_improvement(metric, 0.4, 0.3) == pytest.approx(0.1)
    assert _native_improvement(metric, 0.4, 0.5) == pytest.approx(-0.1)


def test_aggregate_reports_stability_statistics(amazon) -> None:
    manifest, bundle = amazon
    good = score_bundle(manifest, bundle)
    bad_bundle = replace(
        bundle,
        run=replace(bundle.run, run_id="amazon-example-bad", random_seed=12),
        final_test_metrics={**bundle.final_test_metrics, "NDCG@10": 0.180},
    )
    bad = score_bundle(manifest, bad_bundle)
    aggregate = aggregate_scorecards([good, bad])
    assert aggregate.n_runs == 2
    assert aggregate.n_tasks == 1
    assert aggregate.runs_per_task == {"amazon-seqrec-v0": 2}
    assert aggregate.probability_of_meaningful_improvement == pytest.approx(0.5)
    assert aggregate.mean_normalized_improvement == pytest.approx((0.023 - 0.008) / 2)
    assert aggregate.cvar20_normalized_improvement == pytest.approx(-0.008)
    assert aggregate.probability_of_verified_self_evolution == pytest.approx(0.5)
    assert aggregate.mean_productive_trial_ratio == pytest.approx(1.0)


def test_aggregate_weights_tasks_equally_even_with_unequal_seed_counts(amazon) -> None:
    manifest, bundle = amazon
    amazon_card = score_bundle(manifest, bundle)
    amazon_second = replace(amazon_card, run_id="amazon-2")
    other_task = replace(
        amazon_card,
        task_id="another-task",
        run_id="other-1",
        normalized_improvement=-0.01,
        meaningful_improvement=False,
    )
    aggregate = aggregate_scorecards([amazon_card, amazon_second, other_task])
    assert aggregate.mean_normalized_improvement == pytest.approx((0.023 - 0.01) / 2)
    assert aggregate.probability_of_meaningful_improvement == pytest.approx(0.5)
