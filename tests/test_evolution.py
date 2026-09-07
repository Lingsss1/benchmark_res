import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from recevobench.contracts import ContractError, RunTrace
from recevobench.evolution import analyze_code_exploration, analyze_evolution_process
from recevobench.io import load_evaluation_bundle, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def _example():
    return (
        load_manifest(ROOT / "tasks" / "amazon-seqrec-v0.json"),
        load_evaluation_bundle(ROOT / "examples" / "amazon-evaluation.example.json"),
    )


def test_feedback_improvement_that_regresses_hidden_is_not_conversion() -> None:
    manifest, bundle = _example()
    trials = list(bundle.run.trials)
    trials[2] = replace(trials[2], public_metric=0.230, hidden_select_metric=0.190)
    changed = replace(bundle, run=replace(bundle.run, trials=tuple(trials)))
    metrics = analyze_evolution_process(manifest, changed)
    assert metrics.feedback_conversion_rate == pytest.approx(0.5)
    assert metrics.misleading_feedback_rate == pytest.approx(0.5)
    assert metrics.regression_recovery_rate == pytest.approx(0.0)


def test_candidate_parent_must_reference_an_earlier_trial() -> None:
    _, bundle = _example()
    payload = asdict(bundle.run)
    payload["trials"][0]["parent_id"] = "cand-003"

    with pytest.raises(ContractError, match="non-earlier parent"):
        RunTrace.from_dict(payload)


def test_code_exploration_uses_structural_features(tmp_path) -> None:
    manifest, bundle = _example()
    root = tmp_path / "campaign"
    seed = root / "seed-workspace"
    shutil.copytree(ROOT / "seed" / "simple_rec", seed)
    first = root / "candidates" / "cand-001"
    final = root / "candidates" / "cand-003"
    shutil.copytree(seed, first)
    shutil.copytree(seed, final)
    with (final / "model.py").open("a", encoding="utf-8") as handle:
        handle.write("\n\ndef new_strategy(value):\n    return value * 2\n")

    metrics = analyze_code_exploration(manifest, bundle, root)
    assert metrics.code_exploration_reach is not None
    assert metrics.code_exploration_reach > 0
    assert metrics.code_exploration_spread == pytest.approx(metrics.code_exploration_reach)
    assert metrics.structural_unique_candidate_ratio == pytest.approx(1.0)
