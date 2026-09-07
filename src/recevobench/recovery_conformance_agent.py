"""Reference repair agent used only to verify the recovery protocol end to end."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .campaign import evaluate_trial
from .control import (
    finalize_remote,
    submit_remote_recovery_diagnosis,
    submit_remote_trial,
)
from .recovery import RecoveryError, load_fault_catalog


def _detect_and_repair(workspace: Path) -> str:
    matches: list[str] = []
    catalog = load_fault_catalog()
    for fault in catalog.values():
        target = workspace / fault.target_file
        if target.is_file() and target.read_text(encoding="utf-8").count(fault.replace) == 1:
            matches.append(fault.fault_id)
    if len(matches) != 1:
        raise RecoveryError(f"expected exactly one injected recovery fault, found {matches}")
    fault = catalog[matches[0]]
    target = workspace / fault.target_file
    content = target.read_text(encoding="utf-8")
    target.write_text(content.replace(fault.replace, fault.find, 1), encoding="utf-8")
    return fault.fault_id


def _record_memory(memory_dir: Path, episode: str, fault_id: str) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / "recovery-history.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"episode": episode, "diagnosis": fault_id}) + "\n")


def main() -> int:
    workspace = Path(os.environ["RECEVOBENCH_WORKSPACE"])
    fault_id = _detect_and_repair(workspace)
    episode = os.environ.get("RECEVOBENCH_RECOVERY_EPISODE", "unknown")
    _record_memory(Path(os.environ["RECEVOBENCH_AGENT_MEMORY"]), episode, fault_id)
    candidate_id = "recovery-repair-001"
    socket_path = os.environ.get("RECEVOBENCH_CONTROL_SOCKET")
    if socket_path:
        submit_remote_recovery_diagnosis(socket_path, [fault_id])
        submit_remote_trial(
            socket_path,
            candidate_id=candidate_id,
            parent_id="seed",
            hypothesis="Restore the detected injected signal defect.",
            promoted=True,
        )
        finalize_remote(socket_path, candidate_id)
    else:
        run_dir = Path(os.environ["RECEVOBENCH_RUN_DIR"])
        from .recovery import record_recovery_diagnosis

        record_recovery_diagnosis(run_dir, [fault_id])
        evaluate_trial(
            run_dir=run_dir,
            candidate_id=candidate_id,
            parent_id="seed",
            hypothesis="Restore the detected injected signal defect.",
            promoted=True,
        )
        (run_dir / "final-candidate.txt").write_text(candidate_id + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
