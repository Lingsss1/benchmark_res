"""Deterministic popularity/transition baseline shared by smoke tasks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

BEHAVIOR_WEIGHT = {
    "click": 1.0,
    "favorite": 2.0,
    "cart": 3.0,
    "purchase": 4.0,
    "review": 1.0,
    "watch": 1.0,
    "interaction": 1.0,
}


def fit(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    popularity: dict[str, float] = defaultdict(float)
    item_sum: dict[str, float] = defaultdict(float)
    item_count: dict[str, int] = defaultdict(int)
    user_sum: dict[str, float] = defaultdict(float)
    user_count: dict[str, int] = defaultdict(int)
    user_history: dict[str, list[tuple[int, str]]] = defaultdict(list)
    global_sum = 0.0
    global_count = 0
    for row in rows:
        user = row["user_id"]
        item = row["item_id"]
        value = float(row.get("value") or 1.0)
        event = row.get("event_type", "interaction")
        weight = BEHAVIOR_WEIGHT.get(event, 1.0)
        popularity[item] += weight
        item_sum[item] += value
        item_count[item] += 1
        user_sum[user] += value
        user_count[user] += 1
        user_history[user].append((int(float(row["timestamp"])), item))
        global_sum += value
        global_count += 1

    transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: dict[str, list[str]] = {}
    last_item: dict[str, str] = {}
    for user, events in user_history.items():
        ordered = [item for _, item in sorted(events)]
        seen[user] = sorted(set(ordered))
        if ordered:
            last_item[user] = ordered[-1]
        for left, right in zip(ordered, ordered[1:], strict=False):
            transitions[left][right] += 1

    global_mean = global_sum / global_count if global_count else 0.0
    return {
        "schema_version": "1.0",
        "global_mean": global_mean,
        "popularity": dict(sorted(popularity.items())),
        "item_mean": {item: item_sum[item] / item_count[item] for item in sorted(item_sum)},
        "user_mean": {user: user_sum[user] / user_count[user] for user in sorted(user_sum)},
        "last_item": dict(sorted(last_item.items())),
        "seen": dict(sorted(seen.items())),
        "transitions": {
            item: dict(sorted(next_items.items()))
            for item, next_items in sorted(transitions.items())
        },
    }


def ranking_score(model: dict[str, Any], user: str, item: str) -> float:
    popularity = float(model.get("popularity", {}).get(item, 0.0))
    last = model.get("last_item", {}).get(user)
    transition = float(model.get("transitions", {}).get(last, {}).get(item, 0.0))
    return math.log1p(popularity) + 2.0 * math.log1p(transition)


def point_score(model: dict[str, Any], user: str, item: str) -> float:
    global_mean = float(model.get("global_mean", 0.0))
    item_mean = float(model.get("item_mean", {}).get(item, global_mean))
    user_mean = float(model.get("user_mean", {}).get(user, global_mean))
    return 0.6 * item_mean + 0.3 * user_mean + 0.1 * global_mean


def probability_from_score(score: float) -> float:
    if score >= 0:
        value = math.exp(-score)
        return 1.0 / (1.0 + value)
    value = math.exp(score)
    return value / (1.0 + value)
