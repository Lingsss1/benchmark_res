from pathlib import Path

import pytest

from recevobench.artifacts import forbidden_changes
from recevobench.campaign import (
    CampaignError,
    evaluate_trial,
    finalize_campaign,
    grade_campaign,
    initialize_campaign,
)
from recevobench.data import PreparationError, prepare_task, verify_dataset_card
from recevobench.io import load_manifest, read_json
from recevobench.ledger import LedgerError, TrialLedger
from recevobench.schema import validate_artifact

ROOT = Path(__file__).resolve().parents[1]


def _prepared_amazon(tmp_path: Path) -> tuple[Path, Path]:
    task = ROOT / "tasks" / "amazon-seqrec-v0.json"
    prepared = tmp_path / "prepared"
    prepare_task(
        load_manifest(task),
        [ROOT / "tests" / "fixtures" / "amazon" / "reviews.jsonl"],
        prepared,
    )
    return task, prepared


def test_full_public_to_sealed_campaign(tmp_path) -> None:
    task, prepared = _prepared_amazon(tmp_path)
    run_dir = tmp_path / "run"
    initialize_campaign(
        task_path=task,
        prepared_dir=prepared,
        run_dir=run_dir,
        agent_id="test-agent",
        random_seed=17,
        timeout_seconds=10,
        allocated_gpus=0,
    )
    event = evaluate_trial(
        run_dir=run_dir,
        candidate_id="cand-001",
        parent_id="seed",
        hypothesis="No-op conformance candidate",
        promoted=True,
    )
    assert event["payload"]["status"] == "valid"
    assert event["payload"]["compute_seconds"] > 0
    finalize_campaign(run_dir=run_dir, final_candidate_id="cand-001")
    bundle, scorecard = grade_campaign(run_dir)
    assert scorecard.absolute_improvement == pytest.approx(0.0)
    assert scorecard.meaningful_improvement is False
    assert bundle.run.trials[0].hidden_select_metric is not None
    validate_artifact("dataset", read_json(prepared / "dataset-card.json"))
    validate_artifact("run", read_json(run_dir / "public-run.json"))
    validate_artifact("scorecard", read_json(run_dir / "scorecard.json"))


def test_dataset_tampering_is_detected(tmp_path) -> None:
    _, prepared = _prepared_amazon(tmp_path)
    with (prepared / "public" / "train.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered,row\n")
    with pytest.raises(PreparationError, match="verification failed"):
        verify_dataset_card(prepared)


def test_non_editable_candidate_change_is_protocol_violation(tmp_path) -> None:
    task, prepared = _prepared_amazon(tmp_path)
    run_dir = tmp_path / "run"
    initialize_campaign(
        task_path=task,
        prepared_dir=prepared,
        run_dir=run_dir,
        agent_id="test-agent",
        random_seed=3,
        timeout_seconds=10,
        allocated_gpus=0,
    )
    (run_dir / "workspace" / "README.md").write_text("not editable", encoding="utf-8")
    event = evaluate_trial(
        run_dir=run_dir,
        candidate_id="bad-candidate",
        parent_id="seed",
        hypothesis="Try to change a protected path",
        promoted=True,
    )
    assert event["payload"]["status"] == "protocol_violation"
    assert "README.md" in event["payload"]["error"]


def test_hash_chain_detects_ledger_rewrite(tmp_path) -> None:
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    ledger.append({"candidate_id": "a"})
    ledger.append({"candidate_id": "b"})
    content = ledger.path.read_text(encoding="utf-8").replace(
        '"candidate_id": "a"', '"candidate_id": "x"'
    )
    ledger.path.write_text(content, encoding="utf-8")
    with pytest.raises(LedgerError, match="hash"):
        ledger.read()


def test_recursive_edit_pattern_includes_root_python_files() -> None:
    before = {"model.py": "old", "config/value.json": "old", "README.md": "old"}
    after = {"model.py": "new", "config/value.json": "new", "README.md": "new"}
    assert forbidden_changes(before, after, ("**/*.py", "config/**")) == ("README.md",)


def test_campaign_rejects_unknown_parent_and_invalid_final(tmp_path) -> None:
    task, prepared = _prepared_amazon(tmp_path)
    run_dir = tmp_path / "run"
    initialize_campaign(
        task_path=task,
        prepared_dir=prepared,
        run_dir=run_dir,
        agent_id="test-agent",
        random_seed=5,
        timeout_seconds=10,
        allocated_gpus=0,
    )
    with pytest.raises(CampaignError, match="unknown parent"):
        evaluate_trial(
            run_dir=run_dir,
            candidate_id="orphan",
            parent_id="missing",
            hypothesis="orphan",
            promoted=False,
        )
    (run_dir / "workspace" / "README.md").write_text("protected", encoding="utf-8")
    evaluate_trial(
        run_dir=run_dir,
        candidate_id="invalid-final",
        parent_id="seed",
        hypothesis="invalid final",
        promoted=False,
    )
    with pytest.raises(CampaignError, match="valid recorded trial"):
        finalize_campaign(run_dir=run_dir, final_candidate_id="invalid-final")


def test_trusted_grading_rejects_candidate_tampering(tmp_path) -> None:
    task, prepared = _prepared_amazon(tmp_path)
    run_dir = tmp_path / "run"
    initialize_campaign(
        task_path=task,
        prepared_dir=prepared,
        run_dir=run_dir,
        agent_id="test-agent",
        random_seed=7,
        timeout_seconds=10,
        allocated_gpus=0,
    )
    evaluate_trial(
        run_dir=run_dir,
        candidate_id="candidate",
        parent_id="seed",
        hypothesis="valid before tampering",
        promoted=True,
    )
    finalize_campaign(run_dir=run_dir, final_candidate_id="candidate")
    (run_dir / "candidates" / "candidate" / "model.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CampaignError, match="snapshot changed"):
        grade_campaign(run_dir)
