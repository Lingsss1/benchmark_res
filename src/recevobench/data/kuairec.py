"""KuaiRec sparse-to-fully-observed preparation."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from itertools import groupby
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


def _resolve_files(raw_paths: Sequence[Path]) -> tuple[Path, Path]:
    if len(raw_paths) == 1 and raw_paths[0].is_dir():
        big = raw_paths[0] / "big_matrix.csv"
        small = raw_paths[0] / "small_matrix.csv"
    elif len(raw_paths) == 2:
        named = {path.name: path for path in raw_paths}
        big = named.get("big_matrix.csv", raw_paths[0])
        small = named.get("small_matrix.csv", raw_paths[1])
    else:
        raise PreparationError("KuaiRec expects a data directory or big/small matrix files")
    if not big.is_file() or not small.is_file():
        raise PreparationError("KuaiRec big_matrix.csv and small_matrix.csv are required")
    return big, small


def iter_matrix(path: Path, source: str) -> Iterator[Interaction]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                duration = float(row.get("video_duration") or 0.0)
                play = float(row.get("play_duration") or 0.0)
                ratio = float(row.get("watch_ratio") or (play / duration if duration else 0.0))
                timestamp = int(float(row.get("timestamp") or row.get("date") or row_number))
                yield Interaction(
                    user_id=str(row["user_id"]),
                    item_id=str(row.get("video_id") or row["item_id"]),
                    timestamp=timestamp,
                    event_type="watch",
                    value=ratio,
                    duration=duration,
                    source=source,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparationError(f"invalid KuaiRec row {path.name}:{row_number}") from exc


class KuaiRecAdapter(DatasetAdapter):
    name = "kuairec"

    def prepare(
        self,
        *,
        task_id: str,
        raw_paths: Sequence[Path],
        output_dir: Path,
        limit: int | None = None,
    ) -> DatasetCard:
        big, small = _resolve_files(raw_paths)
        temporary, store = self.staging_store()
        try:
            big_rows = store.add(iter_matrix(big, "big_matrix"), limit=limit)
            small_rows = store.add(iter_matrix(small, "small_matrix"), limit=limit)
            store.index()
            writer = PreparedWriter(
                output_dir,
                {
                    "train_big_matrix": "public",
                    "feedback_dev_big_matrix": "public",
                    "hidden_select_big_matrix": "hidden",
                    "sealed_test_small_matrix": "sealed",
                },
            )
            try:
                rows = store.iter_rows(where="source = ?", params=("big_matrix",))
                for _, grouped in groupby(rows, key=lambda row: row.user_id):
                    history = list(grouped)
                    if len(history) < 3:
                        continue
                    for interaction in history[:-2]:
                        writer.write("train_big_matrix", interaction)
                    writer.write("feedback_dev_big_matrix", history[-2])
                    writer.write("hidden_select_big_matrix", history[-1])
                for interaction in store.iter_rows(where="source = ?", params=("small_matrix",)):
                    writer.write("sealed_test_small_matrix", interaction)
            finally:
                writer.close()
            catalog = write_catalog(output_dir, store.catalog())
            card = DatasetCard(
                schema_version="1.0",
                task_id=task_id,
                adapter=self.name,
                adapter_version=self.version,
                sources=source_descriptors((big, small)),
                splits=writer.artifacts(),
                catalog=catalog,
                statistics={
                    "big_matrix_rows": big_rows,
                    "small_matrix_rows": small_rows,
                    "users": store.distinct_count("user_id"),
                    "items": store.distinct_count("item_id"),
                },
                license={
                    "spdx": "CC-BY-SA-4.0",
                    "redistributed": "false",
                    "notice": "Prepared from user-supplied official KuaiRec files.",
                },
                protocol={
                    "development_source": "big_matrix",
                    "sealed_source": "small_matrix",
                    "target": "watch_ratio=play_duration/video_duration",
                    "big_matrix_holdout": "last-two-per-user",
                },
            )
            return finalize_card(output_dir, card)
        finally:
            store.close()
            temporary.cleanup()
