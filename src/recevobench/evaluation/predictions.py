"""Prediction artifact parsing and validation."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


class PredictionError(ValueError):
    pass


@dataclass(frozen=True)
class PointPrediction:
    score: float
    probability: float | None = None


def load_rankings(path: str | Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                user = str(row["user_id"])
                items = [str(item) for item in row["items"]]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise PredictionError(f"invalid ranking at line {line_number}") from exc
            if user in rankings:
                raise PredictionError(f"duplicate ranking for user {user!r}")
            if len(items) != len(set(items)):
                raise PredictionError(f"duplicate items in ranking for user {user!r}")
            rankings[user] = items
    return rankings


def load_point_predictions(path: str | Path) -> dict[tuple[str, str], PointPrediction]:
    predictions: dict[tuple[str, str], PointPrediction] = {}
    for user, item, prediction in iter_point_predictions(path):
        key = (user, item)
        if key in predictions:
            raise PredictionError(f"duplicate point prediction for {key!r}")
        predictions[key] = prediction
    return predictions


def iter_point_predictions(
    path: str | Path,
) -> Iterator[tuple[str, str, PointPrediction]]:
    """Validate and stream point predictions without retaining the file in memory."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                user = str(row["user_id"])
                item = str(row["item_id"])
                score = float(row["score"])
                raw_probability = row.get("probability", "")
                probability = None if raw_probability in (None, "") else float(raw_probability)
            except (KeyError, TypeError, ValueError) as exc:
                raise PredictionError(f"invalid point prediction at row {row_number}") from exc
            if not math.isfinite(score) or (
                probability is not None
                and (not math.isfinite(probability) or not 0.0 <= probability <= 1.0)
            ):
                raise PredictionError(f"non-finite or invalid prediction at row {row_number}")
            yield user, item, PointPrediction(score, probability)
