"""Taobao UserBehavior chronological multi-behavior preparation."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path

from .base import (
    DatasetAdapter,
    DatasetCard,
    Interaction,
    PreparationError,
    PreparedWriter,
    finalize_card,
    source_descriptors,
    write_catalog,
)

BEHAVIOR_NAMES = {
    "pv": "click",
    "click": "click",
    "fav": "favorite",
    "favorite": "favorite",
    "cart": "cart",
    "buy": "purchase",
    "purchase": "purchase",
}


def iter_taobao(path: Path) -> Iterator[Interaction]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        has_header = "user" in sample.lower() and "item" in sample.lower()
        if has_header:
            rows = csv.DictReader(handle)
        else:
            rows = (
                dict(
                    zip(
                        ("user_id", "item_id", "category_id", "behavior_type", "timestamp"),
                        row,
                        strict=True,
                    )
                )
                for row in csv.reader(handle)
            )
        for row_number, row in enumerate(rows, start=2 if has_header else 1):
            try:
                behavior = BEHAVIOR_NAMES[str(row["behavior_type"]).lower()]
                yield Interaction(
                    user_id=str(row["user_id"]),
                    item_id=str(row["item_id"]),
                    timestamp=int(float(row["timestamp"])),
                    event_type=behavior,
                    value=1.0,
                    category_id=str(row.get("category_id", "")),
                    source="taobao_userbehavior",
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparationError(f"invalid Taobao row {path.name}:{row_number}") from exc


def _cutoffs(minimum: int, maximum: int) -> tuple[int, int, int, str]:
    official = (1512086400, 1512172800, 1512259200)
    if minimum < official[0] and maximum >= official[2]:
        return (*official, "fixed-2017-12-01/02/03-UTC")
    width = max(4, maximum - minimum + 1)
    return (
        minimum + int(width * 0.70),
        minimum + int(width * 0.80),
        minimum + int(width * 0.90),
        "deterministic-70/80/90-percent-time-range",
    )


class TaobaoAdapter(DatasetAdapter):
    name = "taobao"

    def prepare(
        self,
        *,
        task_id: str,
        raw_paths: Sequence[Path],
        output_dir: Path,
        limit: int | None = None,
    ) -> DatasetCard:
        if len(raw_paths) != 1 or not raw_paths[0].is_file():
            raise PreparationError("Taobao preparation expects one UserBehavior CSV file")
        temporary, store = self.staging_store()
        try:
            raw_rows = store.add(iter_taobao(raw_paths[0]), limit=limit)
            store.index()
            minimum, maximum = store.min_max_timestamp()
            dev_start, hidden_start, sealed_start, cutoff_policy = _cutoffs(minimum, maximum)
            writer = PreparedWriter(
                output_dir,
                {
                    "train_early_window": "public",
                    "feedback_dev_window": "public",
                    "hidden_select_window": "hidden",
                    "sealed_test_window": "sealed",
                },
            )
            try:
                for interaction in store.iter_rows(order_by="timestamp, seq"):
                    if interaction.timestamp < dev_start:
                        split = "train_early_window"
                    elif interaction.timestamp < hidden_start:
                        split = "feedback_dev_window"
                    elif interaction.timestamp < sealed_start:
                        split = "hidden_select_window"
                    else:
                        split = "sealed_test_window"
                    writer.write(split, interaction)
            finally:
                writer.close()
            catalog = write_catalog(output_dir, store.catalog())
            card = DatasetCard(
                schema_version="1.0",
                task_id=task_id,
                adapter=self.name,
                adapter_version=self.version,
                sources=source_descriptors(raw_paths),
                splits=writer.artifacts(),
                catalog=catalog,
                statistics={
                    "raw_rows": raw_rows,
                    "users": store.distinct_count("user_id"),
                    "items": store.distinct_count("item_id"),
                    "minimum_timestamp": minimum,
                    "maximum_timestamp": maximum,
                },
                license={
                    "redistributed": "false",
                    "notice": (
                        "Users must download Taobao UserBehavior from Tianchi and accept its terms."
                    ),
                },
                protocol={
                    "split": "global-chronological-windows",
                    "cutoff_policy": cutoff_policy,
                    "feedback_dev_start": dev_start,
                    "hidden_select_start": hidden_start,
                    "sealed_test_start": sealed_start,
                    "positive_target": "purchase",
                },
            )
            return finalize_card(output_dir, card)
        finally:
            store.close()
            temporary.cleanup()
