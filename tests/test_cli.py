import json
from pathlib import Path

from recevobench.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_validate_public_run_cli(capsys) -> None:
    result = main(
        [
            "validate-run",
            "--task",
            str(ROOT / "tasks" / "amazon-seqrec-v0.json"),
            "--run",
            str(ROOT / "examples" / "amazon-public-run.example.json"),
        ]
    )
    assert result == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_score_cli_writes_machine_readable_artifact(tmp_path) -> None:
    output = tmp_path / "scorecard.json"
    result = main(
        [
            "score",
            "--task",
            str(ROOT / "tasks" / "amazon-seqrec-v0.json"),
            "--evaluation",
            str(ROOT / "examples" / "amazon-evaluation.example.json"),
            "--output",
            str(output),
        ]
    )
    assert result == 0
    assert json.loads(output.read_text())["meaningful_improvement"] is True
