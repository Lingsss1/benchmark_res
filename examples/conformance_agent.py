"""Minimal black-box agent used to verify the public campaign protocol.

It intentionally makes no model change.  This is a conformance fixture, not a
recommendation method or benchmark baseline.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    run_dir = Path(os.environ["RECEVOBENCH_RUN_DIR"])
    command = [
        sys.executable,
        "-m",
        "recevobench.cli",
        "trial",
        "--run-dir",
        str(run_dir),
        "--candidate-id",
        "conformance-001",
        "--parent-id",
        "seed",
        "--hypothesis",
        "Verify the benchmark protocol without changing the seed model.",
        "--promoted",
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    (run_dir / "final-candidate.txt").write_text("conformance-001\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
