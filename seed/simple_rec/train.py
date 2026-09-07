"""Train the canonical dependency-free RecEvoBench seed model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from model import fit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--model-out", required=True, type=Path)
    args = parser.parse_args()
    with args.train.open("r", encoding="utf-8", newline="") as handle:
        model = fit(csv.DictReader(handle))
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
