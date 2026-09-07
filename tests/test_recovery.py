import json
import sys
from pathlib import Path

import pytest

from recevobench.artifacts import directory_digest
from recevobench.data import prepare_task
from recevobench.io import load_manifest, read_json
from recevobench.recovery import (
    RecoveryError,
    grade_recovery_campaign,
    inject_fault,
    load_fault_catalog,
    record_recovery_diagnosis,
    run_recovery_agent,
    sample_faults,
    task_faults,
)
from recevobench.runtime import copy_seed_workspace
from recevobench.schema import validate_artifact

ROOT = Path(__file__).resolve().parents[1]


def _fast_amazon_task(tmp_path: Path) -> Path:
    payload = read_json(ROOT / "tasks" / "amazon-seqrec-v0.json")
    payload["runtime"]["latency_repeats"] = 1
    task = tmp_path / "amazon-fast.json"
    task.write_text(json.dumps(payload), encoding="utf-8")
    return task


def test_fault_sampling_is_seeded_unique_and_not_manifest_order() -> None:
    manifest = load_manifest(ROOT / "tasks" / "amazon-seqrec-v0.json")
    first = sample_faults(manifest, random_seed=17, episodes=2)
    second = sample_faults(manifest, random_seed=17, episodes=2)
    assert [fault.fault_id for fault in first] == [fault.fault_id for fault in second]
    assert len({fault.fault_id for fault in first}) == 2
    assert {fault.fault_id for fault in sample_faults(manifest, random_seed=0, episodes=2)} != {
        fault.fault_id for fault in sample_faults(manifest, random_seed=1, episodes=2)
    }
    with pytest.raises(RecoveryError, match="at least two"):
        sample_faults(manifest, random_seed=17, episodes=1)


def test_fault_injection_is_exact_and_reversible(tmp_path) -> None:
    manifest = load_manifest(ROOT / "tasks" / "amazon-seqrec-v0.json")
    source = tmp_path / "source"
    copy_seed_workspace(manifest, source)
    before = directory_digest(source)
    fault = load_fault_catalog()["ranking_signal_inverted"]
    target = tmp_path / "target"
    injected = inject_fault(source, target, fault)
    assert injected != before
    model = target / fault.target_file
    content = model.read_text(encoding="utf-8")
    model.write_text(content.replace(fault.replace, fault.find, 1), encoding="utf-8")
    assert directory_digest(target) == before


def test_every_task_fault_applies_to_its_canonical_workspace(tmp_path) -> None:
    for task_path in sorted((ROOT / "tasks").glob("*.json")):
        manifest = load_manifest(task_path)
        source = tmp_path / manifest.task_id / "source"
        copy_seed_workspace(manifest, source)
        for fault in task_faults(manifest):
            target = tmp_path / manifest.task_id / fault.fault_id
            assert inject_fault(source, target, fault) != directory_digest(source)


def test_recovery_diagnosis_is_one_shot_and_catalog_constrained(tmp_path) -> None:
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "recovery-context.json").write_text(
        json.dumps(
            {
                "fault_label_vocabulary": [
                    "ranking_signal_inverted",
                    "ranking_comparator_reversed",
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = record_recovery_diagnosis(run_dir, ["ranking_comparator_reversed"])
    assert payload["labels"] == ["ranking_comparator_reversed"]
    with pytest.raises(RecoveryError, match="already submitted"):
        record_recovery_diagnosis(run_dir, ["ranking_signal_inverted"])

    other = tmp_path / "other"
    other.mkdir()
    (other / "recovery-context.json").write_text(
        json.dumps({"fault_label_vocabulary": ["ranking_signal_inverted"]}),
        encoding="utf-8",
    )
    with pytest.raises(RecoveryError, match="unavailable"):
        record_recovery_diagnosis(other, ["invented_fault"])


def test_multi_episode_hidden_fault_recovery_end_to_end(tmp_path) -> None:
    task = _fast_amazon_task(tmp_path)
    manifest = load_manifest(task)
    prepared = tmp_path / "prepared"
    prepare_task(
        manifest,
        [ROOT / "tests" / "fixtures" / "amazon" / "reviews.jsonl"],
        prepared,
    )
    run_dir = tmp_path / "recovery"
    public = run_recovery_agent(
        task_path=task,
        prepared_dir=prepared,
        run_dir=run_dir,
        agent_id="repair-conformance",
        random_seed=23,
        command=[sys.executable, "-m", "recevobench.recovery_conformance_agent"],
        episodes=2,
        timeout_seconds=10,
        allocated_gpus=0,
    )
    assert "fault_schedule" not in public
    public_text = (run_dir / "recovery-public.json").read_text(encoding="utf-8")
    assert "fault_id" not in public_text
    assert "random_seed" not in public_text
    assert not list((run_dir / "episodes").glob("*/fault.json"))

    card = grade_recovery_campaign(run_dir)
    assert card.n_episodes == 2
    assert card.n_scorable_episodes == 2
    assert card.mean_recovery_fraction == pytest.approx(1.0)
    assert card.full_recovery_rate == pytest.approx(1.0)
    assert card.diagnosis_top1_accuracy == pytest.approx(1.0)
    assert card.catastrophic_forgetting_rate == pytest.approx(0.0)
    assert card.self_evolution_recovery_verified is True
    history = (run_dir / "agent-memory" / "recovery-history.jsonl").read_text(encoding="utf-8")
    assert len(history.splitlines()) == 2
    validate_artifact("recovery-scorecard", read_json(run_dir / "recovery-scorecard.json"))
