"""Generate canonical ranking or point-prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from model import point_score, probability_from_score, ranking_score


def load_catalog(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(row["item_id"]) for row in csv.DictReader(handle)]


def input_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [(str(row["user_id"]), str(row["item_id"])) for row in csv.DictReader(handle)]


def write_point(model: dict, pairs: list[tuple[str, str]], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "score", "probability"])
        for user, item in pairs:
            score = point_score(model, user, item)
            writer.writerow(
                [
                    user,
                    item,
                    format(score, ".17g"),
                    format(probability_from_score(score), ".17g"),
                ]
            )


def write_ranking(
    model: dict,
    pairs: list[tuple[str, str]],
    catalog: list[str],
    output_dir: Path,
    top_k: int,
    write_point_scores: bool,
    include_seen: bool,
) -> None:
    users = sorted({user for user, _ in pairs})
    seen = {user: set(items) for user, items in model.get("seen", {}).items()}
    with (output_dir / "ranking.jsonl").open("w", encoding="utf-8") as handle:
        for user in users:
            candidates = [
                item for item in catalog if include_seen or item not in seen.get(user, set())
            ]
            candidates.sort(key=lambda item: (-ranking_score(model, user, item), item))
            handle.write(
                json.dumps(
                    {"user_id": user, "items": candidates[:top_k]},
                    separators=(",", ":"),
                )
                + "\n"
            )
    if write_point_scores:
        with (output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["user_id", "item_id", "score", "probability"])
            for user in users:
                for item in catalog:
                    if not include_seen and item in seen.get(user, set()):
                        continue
                    score = ranking_score(model, user, item)
                    writer.writerow(
                        [
                            user,
                            item,
                            format(score, ".17g"),
                            format(probability_from_score(score), ".17g"),
                        ]
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ranking", "point"), required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--write-point-scores", action="store_true")
    parser.add_argument("--include-seen", action="store_true")
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    pairs = input_pairs(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "point":
        write_point(model, pairs, args.output_dir / "scores.csv")
    else:
        if args.catalog is None:
            parser.error("--catalog is required in ranking mode")
        write_ranking(
            model,
            pairs,
            load_catalog(args.catalog),
            args.output_dir,
            args.top_k,
            args.write_point_scores,
            args.include_seen,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
