"""Task-aware evaluation over normalized prepared datasets."""

from __future__ import annotations

import csv
import math
import sqlite3
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import groupby
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from ..contracts import TaskManifest
from ..data.base import Interaction, iter_interactions
from .metrics import (
    binary_log_loss,
    grouped_auc,
    ranking_metrics,
    spearman_correlation,
)
from .predictions import iter_point_predictions, load_point_predictions, load_rankings


class EvaluationError(ValueError):
    pass


def _catalog_size(prepared_dir: Path) -> int:
    path = prepared_dir / "public" / "catalog.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _split_path(prepared_dir: Path, split_name: str) -> Path:
    public = prepared_dir / "public" / f"{split_name}.csv"
    private = prepared_dir / "private" / f"{split_name}.csv"
    if public.is_file():
        return public
    if private.is_file():
        return private
    raise EvaluationError(f"prepared split not found: {split_name}")


def _positive_map(
    interactions: Iterable[Interaction], event_type: str | None = None
) -> dict[str, set[str]]:
    positives: dict[str, set[str]] = defaultdict(set)
    for interaction in interactions:
        if event_type is None or interaction.event_type == event_type:
            positives[interaction.user_id].add(interaction.item_id)
    return positives


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _rank_task_metrics(
    manifest: TaskManifest,
    prepared_dir: Path,
    split_name: str,
    prediction_dir: Path,
) -> dict[str, float]:
    interactions = list(iter_interactions(_split_path(prepared_dir, split_name)))
    purchase_task = "purchase" in manifest.task_type
    positives = _positive_map(interactions, "purchase" if purchase_task else None)
    rankings = load_rankings(prediction_dir / "ranking.jsonl")
    metrics = ranking_metrics(
        rankings,
        positives,
        catalog_size=_catalog_size(prepared_dir),
        k_values=(10, 20),
    )
    if not purchase_task:
        return metrics

    score_path = prediction_dir / "scores.csv"
    if not score_path.is_file():
        raise EvaluationError("Taobao evaluation requires scores.csv for GAUC and LogLoss")
    points = load_point_predictions(score_path)
    labels_by_user: dict[str, list[int]] = defaultdict(list)
    scores_by_user: dict[str, list[float]] = defaultdict(list)
    probabilities: list[float] = []
    labels: list[int] = []
    for (user, item), prediction in sorted(points.items()):
        label = int(item in positives.get(user, set()))
        labels_by_user[user].append(label)
        scores_by_user[user].append(prediction.score)
        labels.append(label)
        probabilities.append(
            prediction.probability
            if prediction.probability is not None
            else _sigmoid(prediction.score)
        )
    gauc = grouped_auc(labels_by_user, scores_by_user, weight_by="examples")
    if gauc is None:
        raise EvaluationError(
            "PurchaseGAUC has no users with both positive and negative candidates"
        )
    metrics["PurchaseGAUC"] = gauc
    metrics["LogLoss"] = binary_log_loss(labels, probabilities)
    metrics["PurchaseNDCG@10"] = metrics["NDCG@10"]
    metrics["PurchaseRecall@10"] = metrics["Recall@10"]

    click_positives = _positive_map(interactions, "click")
    click_labels_by_user: dict[str, list[int]] = defaultdict(list)
    for user, item in sorted(points):
        click_labels_by_user[user].append(int(item in click_positives.get(user, set())))
    click_gauc = grouped_auc(click_labels_by_user, scores_by_user, weight_by="examples")
    if click_gauc is not None:
        metrics["ClickGAUC"] = click_gauc

    train_name = "train_early_window"
    frequencies = Counter(
        row.item_id for row in iter_interactions(_split_path(prepared_dir, train_name))
    )
    ordered_items = sorted(frequencies, key=lambda item: (frequencies[item], item))
    tail_count = max(1, math.ceil(len(ordered_items) * 0.2))
    tail = set(ordered_items[:tail_count])
    tail_positives = {user: items & tail for user, items in positives.items() if items & tail}
    metrics["TailPurchaseRecall@10"] = (
        ranking_metrics(
            rankings,
            tail_positives,
            catalog_size=_catalog_size(prepared_dir),
            k_values=(10,),
        )["Recall@10"]
        if tail_positives
        else 0.0
    )
    return metrics


def _watch_task_metrics(
    prepared_dir: Path,
    split_name: str,
    prediction_dir: Path,
) -> dict[str, float]:
    with TemporaryDirectory(prefix="recevobench-eval-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "evaluation.sqlite3"))
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=FILE;
                CREATE TABLE truth (
                    user_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    target REAL NOT NULL
                );
                CREATE TABLE predictions (
                    user_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    PRIMARY KEY (user_id, item_id)
                ) WITHOUT ROWID;
                """
            )
            truth_batch: list[tuple[str, str, float]] = []
            for row in iter_interactions(_split_path(prepared_dir, split_name)):
                truth_batch.append((row.user_id, row.item_id, row.value))
                if len(truth_batch) == 10_000:
                    connection.executemany("INSERT INTO truth VALUES (?, ?, ?)", truth_batch)
                    truth_batch.clear()
            if truth_batch:
                connection.executemany("INSERT INTO truth VALUES (?, ?, ?)", truth_batch)

            prediction_batch: list[tuple[str, str, float]] = []
            try:
                for user, item, prediction in iter_point_predictions(prediction_dir / "scores.csv"):
                    prediction_batch.append((user, item, prediction.score))
                    if len(prediction_batch) == 10_000:
                        connection.executemany(
                            "INSERT INTO predictions VALUES (?, ?, ?)", prediction_batch
                        )
                        prediction_batch.clear()
                if prediction_batch:
                    connection.executemany(
                        "INSERT INTO predictions VALUES (?, ?, ?)", prediction_batch
                    )
            except sqlite3.IntegrityError as exc:
                raise EvaluationError("scores.csv contains duplicate user-item pairs") from exc
            connection.execute("CREATE INDEX truth_key ON truth(user_id, item_id)")
            connection.commit()

            missing = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM truth t LEFT JOIN predictions p
                    ON t.user_id=p.user_id AND t.item_id=p.item_id WHERE p.score IS NULL
                    """
                ).fetchone()[0]
            )
            extra = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM predictions p LEFT JOIN truth t
                    ON t.user_id=p.user_id AND t.item_id=p.item_id WHERE t.user_id IS NULL
                    """
                ).fetchone()[0]
            )
            if missing or extra:
                raise EvaluationError(
                    f"scores.csv key mismatch: {missing} missing and {extra} extra pairs"
                )
            return _watch_metrics_from_database(connection, split_name)
        finally:
            connection.close()


def _watch_metrics_from_database(
    connection: sqlite3.Connection, split_name: str
) -> dict[str, float]:
    """Compute exact KuaiRec metrics with disk-backed joins and bounded RAM."""
    connection.executescript(
        """
        CREATE TEMP TABLE score_ranks AS
        SELECT score, ROW_NUMBER() OVER (ORDER BY score) AS score_rank
        FROM (
            SELECT DISTINCT p.score AS score
            FROM truth t JOIN predictions p
            ON t.user_id=p.user_id AND t.item_id=p.item_id
        );
        CREATE UNIQUE INDEX score_rank_value ON score_ranks(score);
        """
    )
    rank_count = int(connection.execute("SELECT COUNT(*) FROM score_ranks").fetchone()[0])
    tree = array("q", [0]) * (rank_count + 1)

    def prefix(index: int) -> int:
        result = 0
        while index > 0:
            result += tree[index]
            index -= index & -index
        return result

    def add(index: int, value: int) -> None:
        while index < len(tree):
            tree[index] += value
            index += index & -index

    rows = connection.execute(
        """
        SELECT t.target, r.score_rank, COUNT(*)
        FROM truth t JOIN predictions p
          ON t.user_id=p.user_id AND t.item_id=p.item_id
        JOIN score_ranks r ON p.score=r.score
        GROUP BY t.target, r.score_rank
        ORDER BY t.target, r.score_rank
        """
    )
    concordant = 0.0
    comparable = 0
    prior = 0
    for _, target_group in groupby(rows, key=lambda row: row[0]):
        cells = [(int(rank), int(count)) for _, rank, count in target_group]
        for rank, count in cells:
            lower = prefix(rank - 1)
            equal = prefix(rank) - lower
            concordant += count * (lower + 0.5 * equal)
            comparable += count * prior
        for rank, count in cells:
            add(rank, count)
            prior += count
    if not comparable:
        raise EvaluationError("XAUC needs at least two distinct watch-ratio targets")

    mae = float(
        connection.execute(
            """
            SELECT AVG(ABS(t.target-p.score)) FROM truth t JOIN predictions p
            ON t.user_id=p.user_id AND t.item_id=p.item_id
            """
        ).fetchone()[0]
    )
    correlations: list[float] = []
    high_watch_ndcg = 0.0
    high_watch_users = 0
    user_rows = connection.execute(
        """
        SELECT t.user_id, t.item_id, t.target, p.score
        FROM truth t JOIN predictions p
        ON t.user_id=p.user_id AND t.item_id=p.item_id
        ORDER BY t.user_id, t.item_id
        """
    )
    for _, grouped_rows in groupby(user_rows, key=lambda row: row[0]):
        grouped = list(grouped_rows)
        targets = [float(row[2]) for row in grouped]
        scores = [float(row[3]) for row in grouped]
        correlation = spearman_correlation(targets, scores)
        if correlation is not None:
            correlations.append(correlation)
        high_watch = {str(row[1]) for row in grouped if float(row[2]) >= 2.0}
        if high_watch:
            ranked = sorted(grouped, key=lambda row: (-float(row[3]), str(row[1])))
            top = [str(row[1]) for row in ranked[:10]]
            hits = [index for index, item in enumerate(top) if item in high_watch]
            dcg = sum(1.0 / math.log2(index + 2) for index in hits)
            ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(high_watch), 10)))
            high_watch_ndcg += dcg / ideal
            high_watch_users += 1
    xauc = concordant / comparable
    metrics = {
        "xAUC": xauc,
        "MAE": mae,
        "UserSpearman": fmean(correlations) if correlations else 0.0,
    }
    if high_watch_users:
        metrics["NDCG@10_HighWatch"] = high_watch_ndcg / high_watch_users
    if "small_matrix" in split_name:
        metrics["FullyObserved_xAUC"] = xauc
    return metrics


def evaluate_prepared_split(
    manifest: TaskManifest,
    prepared_dir: str | Path,
    split_name: str,
    prediction_dir: str | Path,
) -> dict[str, float]:
    """Evaluate one prediction artifact using the task's frozen metric family."""
    prepared = Path(prepared_dir)
    predictions = Path(prediction_dir)
    if "watch_time" in manifest.task_type:
        return _watch_task_metrics(prepared, split_name, predictions)
    return _rank_task_metrics(manifest, prepared, split_name, predictions)
