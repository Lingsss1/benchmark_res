"""Deterministic, dependency-free metrics used by every task grader."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from statistics import fmean
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in order[index:end]:
            ranks[position] = average_rank
        index = end
    return ranks


def binary_auc(labels: Sequence[int | bool], scores: Sequence[float]) -> float | None:
    """ROC AUC with half credit for ties; returns None for one-class input."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if bool(label))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def grouped_auc(
    labels_by_group: Mapping[T, Sequence[int | bool]],
    scores_by_group: Mapping[T, Sequence[float]],
    *,
    weight_by: str = "examples",
) -> float | None:
    """AUC within groups, weighted by eligible examples or equally by group."""
    if weight_by not in {"examples", "groups"}:
        raise ValueError("weight_by must be examples or groups")
    weighted_sum = 0.0
    total_weight = 0.0
    for group in sorted(labels_by_group, key=str):
        labels = labels_by_group[group]
        scores = scores_by_group.get(group)
        if scores is None:
            raise ValueError(f"scores missing for group {group!r}")
        auc = binary_auc(labels, scores)
        if auc is None:
            continue
        weight = float(len(labels) if weight_by == "examples" else 1)
        weighted_sum += auc * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else None


class _Fenwick:
    def __init__(self, size: int) -> None:
        self.tree = [0] * (size + 1)

    def add(self, index: int, value: int = 1) -> None:
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def prefix(self, index: int) -> int:
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result


def continuous_xauc(targets: Sequence[float], scores: Sequence[float]) -> float | None:
    """Pairwise concordance for continuous targets, excluding equal-target pairs.

    A pair receives one point when the higher target has a higher prediction and
    half a point when predictions tie.  This is the frozen RecEvoBench XAUC
    definition for watch-time prediction.
    """
    if len(targets) != len(scores):
        raise ValueError("targets and scores must have equal length")
    if len(targets) < 2:
        return None
    score_values = sorted(set(scores))
    score_rank = {value: index + 1 for index, value in enumerate(score_values)}
    rows = sorted(zip(targets, scores, strict=True), key=lambda pair: pair[0])
    fenwick = _Fenwick(len(score_values))
    concordant = 0.0
    comparable = 0
    prior = 0
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end][0] == rows[start][0]:
            end += 1
        for _, score in rows[start:end]:
            rank = score_rank[score]
            lower = fenwick.prefix(rank - 1)
            equal = fenwick.prefix(rank) - lower
            concordant += lower + 0.5 * equal
            comparable += prior
        for _, score in rows[start:end]:
            fenwick.add(score_rank[score])
            prior += 1
        start = end
    return concordant / comparable if comparable else None


def spearman_correlation(targets: Sequence[float], scores: Sequence[float]) -> float | None:
    if len(targets) != len(scores):
        raise ValueError("targets and scores must have equal length")
    if len(targets) < 2:
        return None
    left = _average_ranks(targets)
    right = _average_ranks(scores)
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def ranking_metrics(
    recommendations: Mapping[str, Sequence[str]],
    positives: Mapping[str, set[str]],
    *,
    catalog_size: int,
    k_values: Iterable[int] = (10, 20),
) -> dict[str, float]:
    """Macro user-level Recall/NDCG/MRR plus catalog coverage."""
    ks = sorted(set(int(k) for k in k_values))
    if not ks or ks[0] <= 0:
        raise ValueError("k values must be positive")
    eligible = sorted(user for user, items in positives.items() if items)
    if not eligible:
        raise ValueError("ranking ground truth has no positive users")
    sums: Counter[str] = Counter()
    covered: dict[int, set[str]] = {k: set() for k in ks}
    for user in eligible:
        ranked = list(recommendations.get(user, ()))
        if len(ranked) != len(set(ranked)):
            raise ValueError(f"duplicate recommended items for user {user!r}")
        truth = positives[user]
        for k in ks:
            top = ranked[:k]
            hits = [index for index, item in enumerate(top) if item in truth]
            recall = len(hits) / len(truth)
            dcg = sum(1.0 / math.log2(index + 2) for index in hits)
            ideal_count = min(len(truth), k)
            ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
            reciprocal = 1.0 / (hits[0] + 1) if hits else 0.0
            sums[f"Recall@{k}"] += recall
            sums[f"NDCG@{k}"] += dcg / ideal if ideal else 0.0
            sums[f"MRR@{k}"] += reciprocal
            covered[k].update(top)
    metrics = {name: value / len(eligible) for name, value in sums.items()}
    for k in ks:
        metrics[f"CatalogCoverage@{k}"] = len(covered[k]) / catalog_size if catalog_size else 0.0
    return metrics


def mean_absolute_error(targets: Sequence[float], predictions: Sequence[float]) -> float:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("MAE needs equally sized non-empty inputs")
    return fmean(
        abs(target - prediction) for target, prediction in zip(targets, predictions, strict=True)
    )


def binary_log_loss(labels: Sequence[int | bool], probabilities: Sequence[float]) -> float:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("log loss needs equally sized non-empty inputs")
    epsilon = 1e-15
    total = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        clipped = min(1.0 - epsilon, max(epsilon, probability))
        total -= math.log(clipped if bool(label) else 1.0 - clipped)
    return total / len(labels)
