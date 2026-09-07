"""Lifecycle for public evolution runs and trusted sealed grading."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifacts import directory_digest, file_hashes, forbidden_changes
from .assets import resolve_task_path
from .contracts import EvaluationBundle, RunTrace, TaskManifest
from .data import verify_dataset_card
from .data.base import sha256_file
from .io import load_manifest, read_json, write_json
from .ledger import TrialLedger
from .runtime import (
    CandidateResult,
    copy_seed_workspace,
    environment_fingerprint,
    execute_candidate,
    feedback_split,
)
from .sandbox import build_agent_docker_command, docker_available, execute_candidate_docker
from .scoring import Scorecard, score_bundle
from .sealing import seal_evaluation, validate_public_run


class CampaignError(ValueError):
    pass


def _safe_id(value: str, field: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not value or any(character not in allowed for character in value):
        raise CampaignError(f"{field} must contain only letters, numbers, dot, underscore or dash")
    return value


def _dataset_digest(prepared_dir: Path) -> str:
    card = prepared_dir / "dataset-card.json"
    if not card.is_file():
        raise CampaignError("prepared dataset is missing dataset-card.json")
    return directory_digest(prepared_dir, ignore=())


def initialize_campaign(
    *,
    task_path: str | Path,
    prepared_dir: str | Path,
    run_dir: str | Path,
    agent_id: str,
    random_seed: int,
    timeout_seconds: float = 3600.0,
    allocated_gpus: int = 1,
    sandbox_image: str | None = None,
) -> dict[str, Any]:
    task = resolve_task_path(task_path)
    prepared = Path(prepared_dir).resolve()
    destination = Path(run_dir).resolve()
    if destination.exists():
        raise CampaignError(f"run directory already exists: {destination}")
    destination.mkdir(parents=True)
    manifest = load_manifest(task)
    verify_dataset_card(prepared)
    _safe_id(agent_id, "agent_id")
    seed_workspace = destination / "seed-workspace"
    workspace = destination / "workspace"
    copy_seed_workspace(manifest, seed_workspace)
    shutil.copytree(seed_workspace, workspace)
    baseline_output = destination / "baseline" / "feedback"
    execution_options = {
        "prepared_dir": prepared,
        "workspace": seed_workspace,
        "split_name": feedback_split(manifest),
        "output_dir": baseline_output,
        "timeout_seconds": timeout_seconds,
        "random_seed": random_seed,
        "allocated_gpus": allocated_gpus,
    }
    if sandbox_image:
        baseline = execute_candidate_docker(
            manifest,
            task_path=task,
            image=sandbox_image,
            trusted_data=False,
            **execution_options,
        )
    else:
        baseline = execute_candidate(manifest, **execution_options)
    if baseline.status != "valid":
        raise CampaignError(f"canonical seed failed calibration: {baseline.error}")
    primary = manifest.primary_metric.name
    if primary not in baseline.metrics:
        raise CampaignError(f"seed feedback metrics miss primary metric {primary}")
    baseline_compute = max(1e-9, baseline.compute_seconds)
    lock = {
        "schema_version": "1.0",
        "run_id": f"{manifest.task_id}-{agent_id}-{random_seed}",
        "task_id": manifest.task_id,
        "agent_id": agent_id,
        "random_seed": random_seed,
        "created_unix_seconds": time.time(),
        "task_path": str(task),
        "task_sha256": sha256_file(task),
        "prepared_dir": str(prepared),
        "dataset_sha256": _dataset_digest(prepared),
        "seed_sha256": directory_digest(seed_workspace),
        "seed_files": file_hashes(seed_workspace),
        "baseline_compute_seconds": baseline_compute,
        "baseline_public_metric": baseline.metrics[primary],
        "allocated_gpus": allocated_gpus,
        "trial_timeout_seconds": timeout_seconds,
        "sandbox_image": sandbox_image,
        "budget": {
            "max_compute_units": manifest.budget.max_compute_units,
            "max_trials": manifest.budget.max_trials,
            "max_wall_seconds": manifest.budget.max_wall_seconds,
            "max_tokens": manifest.budget.max_tokens,
        },
        "environment": environment_fingerprint(),
    }
    write_json(destination / "campaign-lock.json", lock)
    TrialLedger(destination / "trials.jsonl")
    return lock


def _load_campaign(run_dir: Path) -> tuple[dict[str, Any], TaskManifest, TrialLedger]:
    lock = read_json(run_dir / "campaign-lock.json")
    task_reference = (
        lock["task_path"] if Path(str(lock["task_path"])).is_file() else str(lock["task_id"])
    )
    manifest = load_manifest(task_reference)
    if manifest.task_id != lock.get("task_id"):
        raise CampaignError("campaign task identity mismatch")
    ledger = TrialLedger(run_dir / "trials.jsonl")
    ledger.read()
    return lock, manifest, ledger


def _copy_candidate(workspace: Path, target: Path) -> None:
    def ignore(_: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"__pycache__", ".pytest_cache", "artifacts", "predictions"}
        }

    shutil.copytree(workspace, target, ignore=ignore)


def _execute_campaign_candidate(
    lock: dict[str, Any],
    manifest: TaskManifest,
    *,
    workspace: Path,
    split_name: str,
    output_dir: Path,
    trusted_data: bool,
) -> CandidateResult:
    options = {
        "prepared_dir": os.environ.get("RECEVOBENCH_PREPARED_DIR", str(lock["prepared_dir"])),
        "workspace": workspace,
        "split_name": split_name,
        "output_dir": output_dir,
        "timeout_seconds": float(lock["trial_timeout_seconds"]),
        "random_seed": int(lock["random_seed"]),
        "allocated_gpus": int(lock["allocated_gpus"]),
    }
    image = lock.get("sandbox_image")
    if image:
        return execute_candidate_docker(
            manifest,
            task_path=lock["task_path"],
            image=str(image),
            trusted_data=trusted_data,
            **options,
        )
    return execute_candidate(manifest, **options)


def evaluate_trial(
    *,
    run_dir: str | Path,
    candidate_id: str,
    parent_id: str | None,
    hypothesis: str,
    promoted: bool,
    input_tokens: int = 0,
    output_tokens: int = 0,
    human_interventions: int = 0,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    candidate_id = _safe_id(candidate_id, "candidate_id")
    if (root / "public-run.json").exists():
        raise CampaignError("campaign is already finalized")
    lock, manifest, ledger = _load_campaign(root)
    existing = list(ledger.payloads())
    if any(event["candidate_id"] == candidate_id for event in existing):
        raise CampaignError(f"duplicate candidate_id: {candidate_id}")
    if len(existing) >= manifest.budget.max_trials:
        raise CampaignError("trial budget exhausted")
    known_parents = {"seed", *(str(event["candidate_id"]) for event in existing)}
    if parent_id is not None and parent_id not in known_parents:
        raise CampaignError(f"unknown parent_id: {parent_id}")
    new_input_tokens = max(0, int(input_tokens))
    new_output_tokens = max(0, int(output_tokens))
    used_tokens = sum(
        int(event.get("input_tokens", 0)) + int(event.get("output_tokens", 0)) for event in existing
    )
    if (
        manifest.budget.max_tokens is not None
        and used_tokens + new_input_tokens + new_output_tokens > manifest.budget.max_tokens
    ):
        raise CampaignError("token budget exhausted")
    used_compute = sum(float(event.get("compute_seconds", 0.0)) for event in existing)
    used_units = used_compute / float(lock["baseline_compute_seconds"])
    if used_units >= manifest.budget.max_compute_units:
        raise CampaignError("compute budget exhausted")
    elapsed = time.time() - float(lock["created_unix_seconds"])
    if manifest.budget.max_wall_seconds is not None and elapsed >= manifest.budget.max_wall_seconds:
        raise CampaignError("wall-time budget exhausted")

    workspace = root / "workspace"
    after = file_hashes(workspace)
    forbidden = forbidden_changes(lock["seed_files"], after, manifest.editable_paths)
    candidate_target = root / "candidates" / candidate_id
    if forbidden:
        result = None
        status = "protocol_violation"
        public_metric = None
        error = "forbidden workspace changes: " + ", ".join(forbidden)
        candidate_target.parent.mkdir(parents=True, exist_ok=True)
        _copy_candidate(workspace, candidate_target)
    else:
        candidate_target.parent.mkdir(parents=True, exist_ok=True)
        _copy_candidate(workspace, candidate_target)
        if file_hashes(workspace) != after or file_hashes(candidate_target) != after:
            result = None
            status = "invalid"
            public_metric = None
            error = "workspace changed while the controller was freezing the candidate"
        else:
            output = root / "trial-artifacts" / f"{len(existing) + 1:04d}-{candidate_id}"
            result = _execute_campaign_candidate(
                lock,
                manifest,
                workspace=candidate_target,
                split_name=feedback_split(manifest),
                output_dir=output,
                trusted_data=False,
            )
            status = result.status
            public_metric = result.metrics.get(manifest.primary_metric.name)
            error = result.error
    payload = {
        "step": len(existing) + 1,
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "status": status,
        "public_metric": public_metric,
        "hidden_select_metric": None,
        "gpu_seconds": 0.0 if result is None else result.gpu_seconds,
        "wall_seconds": 0.0 if result is None else result.wall_seconds,
        "compute_seconds": 0.0 if result is None else result.compute_seconds,
        "cpu_seconds": 0.0 if result is None else result.cpu_seconds,
        "input_tokens": new_input_tokens,
        "output_tokens": new_output_tokens,
        "promoted": bool(promoted and status == "valid"),
        "hypothesis": hypothesis,
        "human_interventions": max(0, int(human_interventions)),
        "candidate_sha256": directory_digest(candidate_target),
        "error": error,
    }
    event = ledger.append(payload)
    write_json(root / "latest-trial.json", event)
    return event


def finalize_campaign(*, run_dir: str | Path, final_candidate_id: str | None = None) -> RunTrace:
    root = Path(run_dir).resolve()
    if (root / "public-run.json").exists():
        raise CampaignError("campaign is already finalized")
    lock, _, ledger = _load_campaign(root)
    payloads = list(ledger.payloads())
    if final_candidate_id is None:
        promoted = [event for event in payloads if event.get("promoted")]
        valid = [event for event in payloads if event.get("status") == "valid"]
        chosen = promoted[-1:] or valid[-1:]
        final_candidate_id = chosen[0]["candidate_id"] if chosen else "seed"
    if final_candidate_id != "seed":
        selected = next(
            (event for event in payloads if event.get("candidate_id") == final_candidate_id),
            None,
        )
        if selected is None or selected.get("status") != "valid":
            raise CampaignError("final candidate must be a valid recorded trial")
    public_payloads = [
        {
            key: event.get(key)
            for key in (
                "step",
                "candidate_id",
                "parent_id",
                "status",
                "public_metric",
                "hidden_select_metric",
                "gpu_seconds",
                "wall_seconds",
                "compute_seconds",
                "input_tokens",
                "output_tokens",
                "promoted",
                "hypothesis",
                "human_interventions",
                "candidate_sha256",
            )
        }
        for event in payloads
    ]
    run = RunTrace.from_dict(
        {
            "run_id": lock["run_id"],
            "task_id": lock["task_id"],
            "agent_id": lock["agent_id"],
            "random_seed": lock["random_seed"],
            "final_candidate_id": final_candidate_id,
            "baseline_compute_seconds": lock["baseline_compute_seconds"],
            "elapsed_wall_seconds": time.time() - float(lock["created_unix_seconds"]),
            "trials": public_payloads,
        }
    )
    validate_public_run(run)
    write_json(root / "public-run.json", run)
    return run


def _sealed_split(manifest: TaskManifest) -> str:
    return next(split.name for split in manifest.splits if split.visibility == "sealed")


def _hidden_split(manifest: TaskManifest) -> str:
    return next(split.name for split in manifest.splits if split.visibility == "hidden")


def _verify_trusted_inputs(
    root: Path, lock: dict[str, Any], manifest: TaskManifest, ledger: TrialLedger
) -> None:
    prepared = Path(str(lock["prepared_dir"]))
    verify_dataset_card(prepared)
    if _dataset_digest(prepared) != lock.get("dataset_sha256"):
        raise CampaignError("prepared dataset changed after campaign initialization")
    task = resolve_task_path(lock["task_path"])
    if sha256_file(task) != lock.get("task_sha256"):
        raise CampaignError("task manifest changed after campaign initialization")
    if directory_digest(root / "seed-workspace") != lock.get("seed_sha256"):
        raise CampaignError("seed workspace changed after campaign initialization")
    for event in ledger.payloads():
        candidate = root / "candidates" / str(event["candidate_id"])
        if directory_digest(candidate) != event.get("candidate_sha256"):
            raise CampaignError(f"candidate snapshot changed after trial: {event['candidate_id']}")


def grade_campaign(run_dir: str | Path) -> tuple[EvaluationBundle, Scorecard]:
    """Trusted side: re-run seed/candidates on hidden data and final on sealed data."""
    root = Path(run_dir).resolve()
    lock, manifest, ledger = _load_campaign(root)
    _verify_trusted_inputs(root, lock, manifest, ledger)
    public_run = RunTrace.from_dict(read_json(root / "public-run.json"))
    validate_public_run(public_run)
    hidden_name = _hidden_split(manifest)
    test_name = _sealed_split(manifest)
    seed_workspace = root / "seed-workspace"

    baseline_hidden = _execute_campaign_candidate(
        lock,
        manifest,
        workspace=seed_workspace,
        split_name=hidden_name,
        output_dir=root / "sealed-artifacts" / "seed-hidden",
        trusted_data=True,
    )
    baseline_test = _execute_campaign_candidate(
        lock,
        manifest,
        workspace=seed_workspace,
        split_name=test_name,
        output_dir=root / "sealed-artifacts" / "seed-test",
        trusted_data=True,
    )
    if baseline_hidden.status != "valid" or baseline_test.status != "valid":
        raise CampaignError("seed failed trusted evaluation")

    hidden_by_candidate: dict[str, float] = {}
    for trial in public_run.trials:
        if trial.status != "valid":
            continue
        result = _execute_campaign_candidate(
            lock,
            manifest,
            workspace=root / "candidates" / trial.candidate_id,
            split_name=hidden_name,
            output_dir=root / "sealed-artifacts" / f"candidate-{trial.candidate_id}-hidden",
            trusted_data=True,
        )
        if result.status != "valid":
            raise CampaignError(
                f"candidate {trial.candidate_id} failed hidden re-evaluation: {result.error}"
            )
        hidden_by_candidate[trial.candidate_id] = result.metrics[manifest.primary_metric.name]

    if public_run.final_candidate_id == "seed":
        final_test = baseline_test
    else:
        final_test = _execute_campaign_candidate(
            lock,
            manifest,
            workspace=root / "candidates" / public_run.final_candidate_id,
            split_name=test_name,
            output_dir=root / "sealed-artifacts" / "final-test",
            trusted_data=True,
        )
        if final_test.status != "valid":
            raise CampaignError(f"final candidate failed sealed test: {final_test.error}")

    primary = manifest.primary_metric.name
    bundle = seal_evaluation(
        public_run,
        hidden_select_by_candidate=hidden_by_candidate,
        baseline_hidden_select_metrics={primary: baseline_hidden.metrics[primary]},
        baseline_test_metrics={primary: baseline_test.metrics[primary]},
        final_test_metrics={primary: final_test.metrics[primary]},
        baseline_guardrail_metrics=baseline_test.metrics,
        final_guardrail_metrics=final_test.metrics,
        baseline_public_metrics={primary: float(lock["baseline_public_metric"])},
    )
    card = score_bundle(manifest, bundle, artifact_root=root)
    write_json(root / "evaluation-bundle.json", bundle)
    write_json(root / "scorecard.json", card)
    return bundle, card


def run_agent_command(
    *,
    task_path: str | Path,
    prepared_dir: str | Path,
    run_dir: str | Path,
    agent_id: str,
    random_seed: int,
    command: list[str],
    timeout_seconds: float,
    allocated_gpus: int = 1,
    sandbox_image: str | None = None,
) -> RunTrace:
    """Initialize a campaign, execute a black-box command, then freeze its run."""
    initialize_campaign(
        task_path=task_path,
        prepared_dir=prepared_dir,
        run_dir=run_dir,
        agent_id=agent_id,
        random_seed=random_seed,
        timeout_seconds=timeout_seconds,
        allocated_gpus=allocated_gpus,
        sandbox_image=sandbox_image,
    )
    root = Path(run_dir).resolve()
    manifest = load_manifest(task_path)
    environment = {
        **os.environ,
        "RECEVOBENCH_RUN_DIR": str(root),
        "RECEVOBENCH_WORKSPACE": str(root / "workspace"),
        "RECEVOBENCH_PUBLIC_DATA": str(Path(prepared_dir).resolve() / "public"),
        "RECEVOBENCH_TASK_PATH": str(Path(task_path).resolve()),
        "RECEVOBENCH_FEEDBACK_SPLIT": feedback_split(manifest),
        "RECEVOBENCH_RANDOM_SEED": str(random_seed),
        "RECEVOBENCH_AGENT_ID": agent_id,
        "RECEVOBENCH_TRIAL_COMMAND_JSON": json.dumps(
            [
                os.sys.executable,
                "-m",
                "recevobench.cli",
                "trial",
                "--run-dir",
                str(root),
            ]
        ),
    }
    maximum_wall = manifest.budget.max_wall_seconds or timeout_seconds
    logs = root / "agent-logs"
    logs.mkdir()
    stdout_path = logs / "stdout.log"
    stderr_path = logs / "stderr.log"
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=root / "workspace",
                env=environment,
                timeout=maximum_wall,
                check=False,
                stdout=stdout,
                stderr=stderr,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
            timed_out = True
    write_json(
        root / "agent-exit.json",
        {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "command": command,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )
    final_path = root / "final-candidate.txt"
    final_id = final_path.read_text(encoding="utf-8").strip() if final_path.is_file() else None
    return finalize_campaign(run_dir=root, final_candidate_id=final_id)


def run_agent_docker(
    *,
    task_path: str | Path,
    prepared_dir: str | Path,
    run_dir: str | Path,
    agent_id: str,
    random_seed: int,
    command: list[str],
    image: str,
    timeout_seconds: float,
    allocated_gpus: int = 1,
    network: str = "none",
    memory: str = "32g",
    cpus: float = 8.0,
) -> RunTrace:
    """Official-mode runner: agent and every candidate execute in separate sandboxes."""
    if not docker_available():
        raise CampaignError("Docker is required for run-agent-docker")
    task = resolve_task_path(task_path)
    prepared = Path(prepared_dir).resolve()
    root = Path(run_dir).resolve()
    manifest = load_manifest(task)
    initialize_campaign(
        task_path=task,
        prepared_dir=prepared,
        run_dir=root,
        agent_id=agent_id,
        random_seed=random_seed,
        timeout_seconds=timeout_seconds,
        allocated_gpus=allocated_gpus,
        sandbox_image=image,
    )
    from .control import ControlServer

    with tempfile.TemporaryDirectory(prefix="recevobench-control-") as temporary:
        control_directory = Path(temporary)
        socket_path = control_directory / "control.sock"
        docker_command = build_agent_docker_command(
            image=image,
            workspace=root / "workspace",
            public_data=prepared / "public",
            task_path=task,
            control_directory=control_directory,
            command=command,
            allocated_gpus=allocated_gpus,
            network=network,
            memory=memory,
            cpus=cpus,
        )
        logs = root / "agent-logs"
        logs.mkdir()
        timed_out = False
        with (
            ControlServer(socket_path, root),
            (logs / "stdout.log").open("wb") as stdout,
            (logs / "stderr.log").open("wb") as stderr,
        ):
            try:
                completed = subprocess.run(
                    docker_command,
                    check=False,
                    timeout=manifest.budget.max_wall_seconds or timeout_seconds,
                    stdout=stdout,
                    stderr=stderr,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
                timed_out = True
        write_json(
            root / "agent-exit.json",
            {
                "exit_code": exit_code,
                "timed_out": timed_out,
                "container_image": image,
                "network": network,
                "command": command,
            },
        )
    final_path = root / "final-candidate.txt"
    final_id = final_path.read_text(encoding="utf-8").strip() if final_path.is_file() else None
    return finalize_campaign(run_dir=root, final_candidate_id=final_id)
