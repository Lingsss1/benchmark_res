import csv
import json
from pathlib import Path

import pytest

from recevobench.data import prepare_task
from recevobench.data.base import iter_interactions
from recevobench.evaluation import (
    binary_auc,
    continuous_xauc,
    evaluate_prepared_split,
    ranking_metrics,
)
from recevobench.io import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_binary_auc_and_continuous_xauc_handle_ties() -> None:
    assert binary_auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx(1.0)
    assert binary_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert continuous_xauc([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]) == pytest.approx(1.0)
    assert continuous_xauc([1.0, 2.0], [0.5, 0.5]) == pytest.approx(0.5)


def test_ranking_metrics_use_macro_users_and_binary_ndcg() -> None:
    metrics = ranking_metrics(
        {"u1": ["a", "b"], "u2": ["b", "a"]},
        {"u1": {"a"}, "u2": {"a"}},
        catalog_size=2,
        k_values=(1, 2),
    )
    assert metrics["Recall@1"] == pytest.approx(0.5)
    assert metrics["NDCG@1"] == pytest.approx(0.5)
    assert metrics["Recall@2"] == pytest.approx(1.0)
    assert metrics["CatalogCoverage@1"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("task_file", "raw_names", "expected_splits"),
    [
        (
            "amazon-seqrec-v0.json",
            ["amazon/reviews.jsonl"],
            {"train": 6, "feedback_dev": 2, "hidden_select": 2, "sealed_test": 2},
        ),
        (
            "kuairec-watchtime-v0.json",
            ["kuairec"],
            {
                "train_big_matrix": 4,
                "feedback_dev_big_matrix": 2,
                "hidden_select_big_matrix": 2,
                "sealed_test_small_matrix": 8,
            },
        ),
        (
            "taobao-purchase-v0.json",
            ["taobao/UserBehavior.csv"],
            {
                "train_early_window": 5,
                "feedback_dev_window": 3,
                "hidden_select_window": 2,
                "sealed_test_window": 3,
            },
        ),
    ],
)
def test_prepare_adapters_are_deterministic(
    tmp_path, task_file, raw_names, expected_splits
) -> None:
    manifest = load_manifest(ROOT / "tasks" / task_file)
    raw = [ROOT / "tests" / "fixtures" / name for name in raw_names]
    card = prepare_task(manifest, raw, tmp_path / "prepared")
    assert {name: artifact.rows for name, artifact in card.splits.items()} == expected_splits
    assert all(len(artifact.sha256) == 64 for artifact in card.splits.values())
    assert len(card.catalog.sha256) == 64
    serialized = json.loads((tmp_path / "prepared" / "dataset-card.json").read_text())
    assert serialized["task_id"] == manifest.task_id


def test_kuairec_perfect_predictions_score_perfectly(tmp_path) -> None:
    manifest = load_manifest(ROOT / "tasks" / "kuairec-watchtime-v0.json")
    prepared = tmp_path / "prepared"
    prepare_task(manifest, [ROOT / "tests" / "fixtures" / "kuairec"], prepared)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    truth = list(iter_interactions(prepared / "private" / "sealed_test_small_matrix.csv"))
    with (predictions / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "score"])
        for row in truth:
            writer.writerow([row.user_id, row.item_id, row.value])
    metrics = evaluate_prepared_split(manifest, prepared, "sealed_test_small_matrix", predictions)
    assert metrics["xAUC"] == pytest.approx(1.0)
    assert metrics["FullyObserved_xAUC"] == pytest.approx(1.0)
    assert metrics["MAE"] == pytest.approx(0.0)


def test_amazon_ranking_evaluator(tmp_path) -> None:
    manifest = load_manifest(ROOT / "tasks" / "amazon-seqrec-v0.json")
    prepared = tmp_path / "prepared"
    prepare_task(
        manifest,
        [ROOT / "tests" / "fixtures" / "amazon" / "reviews.jsonl"],
        prepared,
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    lines = [
        {"user_id": "u1", "items": ["i6", "i1", "i2", "i3", "i4", "i5"]},
        {"user_id": "u2", "items": ["i1", "i2", "i3", "i4", "i5", "i6"]},
    ]
    (predictions / "ranking.jsonl").write_text("".join(json.dumps(row) + "\n" for row in lines))
    metrics = evaluate_prepared_split(manifest, prepared, "sealed_test", predictions)
    assert metrics["NDCG@10"] == pytest.approx(1.0)
    assert metrics["Recall@10"] == pytest.approx(1.0)
