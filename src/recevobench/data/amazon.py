"""Amazon Reviews sequential-recommendation preparation."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Sequence
from itertools import groupby
from pathlib import Path
from typing import Any, TextIO

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


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def iter_amazon(path: Path) -> Iterator[Interaction]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreparationError(f"invalid Amazon JSON at {path.name}:{line_number}") from exc
            user = _first(record, "user_id", "reviewerID")
            item = _first(record, "parent_asin", "asin")
            timestamp = _first(record, "timestamp", "unixReviewTime")
            if user is None or item is None or timestamp is None:
                continue
            timestamp_i = int(float(timestamp))
            if timestamp_i > 100_000_000_000:
                timestamp_i //= 1000
            yield Interaction(
                user_id=str(user),
                item_id=str(item),
                timestamp=timestamp_i,
                event_type="review",
                value=float(_first(record, "rating", "overall") or 1.0),
                category_id=str(_first(record, "category", "main_category") or ""),
                source="amazon_reviews",
            )


class AmazonAdapter(DatasetAdapter):
    name = "amazon"

    def prepare(
        self,
        *,
        task_id: str,
        raw_paths: Sequence[Path],
        output_dir: Path,
        limit: int | None = None,
    ) -> DatasetCard:
        if len(raw_paths) != 1 or not raw_paths[0].is_file():
            raise PreparationError("Amazon preparation expects one JSONL or JSONL.GZ file")
        temporary, store = self.staging_store()
        try:
            raw_rows = store.add(iter_amazon(raw_paths[0]), limit=limit)
            store.index()
            removed = store.prune_k_core(min_user=5, min_item=1)
            writer = PreparedWriter(
                output_dir,
                {
                    "train": "public",
                    "feedback_dev": "public",
                    "hidden_select": "hidden",
                    "sealed_test": "sealed",
                },
            )
            try:
                rows = store.iter_rows()
                for _, grouped in groupby(rows, key=lambda row: row.user_id):
                    history = list(grouped)
                    if len(history) < 5:
                        continue
                    for interaction in history[:-3]:
                        writer.write("train", interaction)
                    writer.write("feedback_dev", history[-3])
                    writer.write("hidden_select", history[-2])
                    writer.write("sealed_test", history[-1])
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
                    "k_core_removed_rows": removed,
                    "users": store.distinct_count("user_id"),
                    "items": store.distinct_count("item_id"),
                },
                license={
                    "redistributed": "false",
                    "notice": (
                        "Users must obtain Amazon Reviews from the official source "
                        "and comply with its terms."
                    ),
                },
                protocol={
                    "split": "per-user-last-three",
                    "minimum_user_interactions": 5,
                    "timestamp_unit": "seconds",
                },
            )
            return finalize_card(output_dir, card)
        finally:
            store.close()
            temporary.cleanup()
