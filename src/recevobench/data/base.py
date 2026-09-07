"""Shared, dependency-free infrastructure for reproducible dataset preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..io import write_json

INTERACTION_FIELDS = (
    "user_id",
    "item_id",
    "timestamp",
    "event_type",
    "value",
    "duration",
    "category_id",
    "source",
)


class PreparationError(ValueError):
    """Raised when raw data cannot be converted without ambiguity or leakage."""


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: int
    event_type: str = "interaction"
    value: float = 1.0
    duration: float | None = None
    category_id: str = ""
    source: str = ""

    def to_row(self) -> dict[str, str | int | float]:
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "value": format(self.value, ".17g"),
            "duration": "" if self.duration is None else format(self.duration, ".17g"),
            "category_id": self.category_id,
            "source": self.source,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Interaction:
        duration = row.get("duration", "")
        return cls(
            user_id=str(row["user_id"]),
            item_id=str(row["item_id"]),
            timestamp=int(float(row["timestamp"])),
            event_type=str(row.get("event_type", "interaction")),
            value=float(row.get("value", 1.0)),
            duration=None if duration in (None, "") else float(duration),
            category_id=str(row.get("category_id", "")),
            source=str(row.get("source", "")),
        )


@dataclass(frozen=True)
class SplitArtifact:
    path: str
    rows: int
    sha256: str


@dataclass(frozen=True)
class DatasetCard:
    schema_version: str
    task_id: str
    adapter: str
    adapter_version: str
    sources: tuple[dict[str, str], ...]
    splits: dict[str, SplitArtifact]
    catalog: SplitArtifact
    statistics: dict[str, int | float | str]
    license: dict[str, str]
    protocol: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = list(self.sources)
        return data


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_descriptors(paths: Sequence[Path]) -> tuple[dict[str, str], ...]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise PreparationError(f"raw source does not exist: {path}")
    return tuple({"name": path.name, "sha256": sha256_file(path)} for path in files)


class InteractionStore:
    """SQLite-backed staging store that scales beyond memory-sized datasets."""

    def __init__(self, root: Path) -> None:
        self.path = root / "interactions.sqlite3"
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute(
            """
            CREATE TABLE interactions (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                value REAL NOT NULL,
                duration REAL,
                category_id TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )

    def add(self, interactions: Iterable[Interaction], limit: int | None = None) -> int:
        rows: list[tuple[Any, ...]] = []
        total = 0
        for interaction in interactions:
            if not interaction.user_id or not interaction.item_id:
                raise PreparationError("user_id and item_id cannot be empty")
            rows.append(
                (
                    interaction.user_id,
                    interaction.item_id,
                    interaction.timestamp,
                    interaction.event_type,
                    interaction.value,
                    interaction.duration,
                    interaction.category_id,
                    interaction.source,
                )
            )
            total += 1
            if len(rows) >= 10_000:
                self._insert(rows)
                rows.clear()
            if limit is not None and total >= limit:
                break
        if rows:
            self._insert(rows)
        self.connection.commit()
        return total

    def _insert(self, rows: list[tuple[Any, ...]]) -> None:
        self.connection.executemany(
            """
            INSERT INTO interactions
              (user_id, item_id, timestamp, event_type, value, duration, category_id, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def index(self) -> None:
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_time ON interactions(user_id, timestamp, seq)"
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_item ON interactions(item_id)")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_time ON interactions(source, timestamp, seq)"
        )
        self.connection.commit()

    def prune_k_core(self, min_user: int, min_item: int) -> int:
        """Iteratively enforce interaction-frequency constraints."""
        deleted_total = 0
        while True:
            before = self.count()
            if min_user > 1:
                self.connection.execute(
                    """
                    DELETE FROM interactions WHERE user_id IN (
                      SELECT user_id FROM interactions GROUP BY user_id HAVING COUNT(*) < ?
                    )
                    """,
                    (min_user,),
                )
            if min_item > 1:
                self.connection.execute(
                    """
                    DELETE FROM interactions WHERE item_id IN (
                      SELECT item_id FROM interactions GROUP BY item_id HAVING COUNT(*) < ?
                    )
                    """,
                    (min_item,),
                )
            self.connection.commit()
            after = self.count()
            deleted_total += before - after
            if before == after:
                return deleted_total

    def count(self, where: str = "", params: Sequence[Any] = ()) -> int:
        clause = f" WHERE {where}" if where else ""
        return int(
            self.connection.execute(
                f"SELECT COUNT(*) FROM interactions{clause}", params
            ).fetchone()[0]
        )

    def distinct_count(self, column: str) -> int:
        if column not in {"user_id", "item_id"}:
            raise ValueError("unsupported distinct-count column")
        return int(
            self.connection.execute(
                f"SELECT COUNT(DISTINCT {column}) FROM interactions"
            ).fetchone()[0]
        )

    def min_max_timestamp(self, source: str | None = None) -> tuple[int, int]:
        if source is None:
            row = self.connection.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM interactions"
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM interactions WHERE source = ?",
                (source,),
            ).fetchone()
        if row[0] is None or row[1] is None:
            raise PreparationError("source contains no usable interactions")
        return int(row[0]), int(row[1])

    def iter_rows(
        self,
        *,
        where: str = "",
        params: Sequence[Any] = (),
        order_by: str = "user_id, timestamp, seq",
    ) -> Iterator[Interaction]:
        allowed_orders = {
            "user_id, timestamp, seq",
            "timestamp, seq",
            "item_id",
        }
        if order_by not in allowed_orders:
            raise ValueError("unsupported ordering")
        clause = f" WHERE {where}" if where else ""
        query = (
            "SELECT user_id, item_id, timestamp, event_type, value, duration, "
            f"category_id, source FROM interactions{clause} ORDER BY {order_by}"
        )
        for row in self.connection.execute(query, params):
            yield Interaction(*row)

    def catalog(self) -> Iterator[str]:
        for (item_id,) in self.connection.execute(
            "SELECT DISTINCT item_id FROM interactions ORDER BY item_id"
        ):
            yield str(item_id)

    def close(self) -> None:
        self.connection.close()


class PreparedWriter:
    def __init__(self, output_dir: Path, split_visibility: Mapping[str, str]) -> None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise PreparationError(
                f"output directory is not empty: {output_dir}; choose a new directory"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.paths: dict[str, Path] = {}
        self.handles: dict[str, Any] = {}
        self.writers: dict[str, csv.DictWriter] = {}
        self.counts = {name: 0 for name in split_visibility}
        for name, visibility in split_visibility.items():
            directory = output_dir / ("public" if visibility == "public" else "private")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{name}.csv"
            handle = path.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=INTERACTION_FIELDS)
            writer.writeheader()
            self.paths[name] = path
            self.handles[name] = handle
            self.writers[name] = writer

    def write(self, split: str, interaction: Interaction) -> None:
        self.writers[split].writerow(interaction.to_row())
        self.counts[split] += 1

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()

    def artifacts(self) -> dict[str, SplitArtifact]:
        return {
            name: SplitArtifact(
                path=str(path.relative_to(self.output_dir)),
                rows=self.counts[name],
                sha256=sha256_file(path),
            )
            for name, path in self.paths.items()
        }


def write_catalog(output_dir: Path, item_ids: Iterable[str]) -> SplitArtifact:
    path = output_dir / "public" / "catalog.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id"])
        for item_id in item_ids:
            writer.writerow([item_id])
            rows += 1
    return SplitArtifact(
        path=str(path.relative_to(output_dir)), rows=rows, sha256=sha256_file(path)
    )


def iter_interactions(path: str | Path) -> Iterator[Interaction]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield Interaction.from_row(row)


def finalize_card(output_dir: Path, card: DatasetCard) -> DatasetCard:
    write_json(output_dir / "dataset-card.json", card)
    return card


def verify_dataset_card(output_dir: str | Path) -> dict[str, Any]:
    """Verify every prepared artifact against the frozen dataset card."""
    root = Path(output_dir)
    card_path = root / "dataset-card.json"
    if not card_path.is_file():
        raise PreparationError("dataset-card.json is missing")
    with card_path.open("r", encoding="utf-8") as handle:
        card = json.load(handle)
    failures: list[str] = []
    artifacts = dict(card.get("splits", {}))
    artifacts["catalog"] = card.get("catalog", {})
    verified: dict[str, dict[str, Any]] = {}
    for name, artifact in artifacts.items():
        relative = artifact.get("path")
        if not isinstance(relative, str):
            failures.append(f"{name}: missing path")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            failures.append(f"{name}: path escapes prepared directory")
            continue
        if not path.is_file():
            failures.append(f"{name}: file missing")
            continue
        actual_hash = sha256_file(path)
        expected_hash = artifact.get("sha256")
        if actual_hash != expected_hash:
            failures.append(f"{name}: checksum mismatch")
        with path.open("r", encoding="utf-8", newline="") as handle:
            actual_rows = max(0, sum(1 for _ in handle) - 1)
        expected_rows = int(artifact.get("rows", -1))
        if actual_rows != expected_rows:
            failures.append(f"{name}: row count {actual_rows} does not match {expected_rows}")
        verified[name] = {"rows": actual_rows, "sha256": actual_hash}
    if failures:
        raise PreparationError("dataset verification failed: " + "; ".join(failures))
    return {
        "ok": True,
        "task_id": card.get("task_id"),
        "artifacts": verified,
    }


class DatasetAdapter(ABC):
    name: str
    version = "1.0"

    @abstractmethod
    def prepare(
        self,
        *,
        task_id: str,
        raw_paths: Sequence[Path],
        output_dir: Path,
        limit: int | None = None,
    ) -> DatasetCard:
        raise NotImplementedError

    def staging_store(self) -> tuple[TemporaryDirectory[str], InteractionStore]:
        temporary = TemporaryDirectory(prefix=f"recevobench-{self.name}-")
        return temporary, InteractionStore(Path(temporary.name))
