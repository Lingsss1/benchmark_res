"""Seeded hidden-fault episodes for evaluating autonomous model recovery."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any

from .artifacts import directory_digest
from .assets import resolve_asset, resolve_task_path
from .campaign import (
    _execute_campaign_candidate,
    _hidden_split,
    _load_campaign,
    _sealed_split,
    finalize_campaign,
    grade_campaign,
    initialize_campaign,
)
from .contracts import EvaluationBundle, MetricSpec, TaskManifest
from .data import verify_dataset_card
from .data.base import sha256_file
from .io import load_manifest, read_json, write_json
from .runtime import copy_seed_workspace, execute_candidate, feedback_split
from .sandbox import build_agent_docker_command, docker_available, execute_candidate_docker
from .scoring import _guardrail_result

FAULT_CATALOG_PATH = "faults/recovery-faults-v1.json"


class RecoveryError(ValueError):
    """Raised when a recovery campaign violates its protocol."""


@dataclass(frozen=True)
class FaultSpec:
    fault_id: str
    family: str
    title: str
    task_types: tuple[str, ...]
    target_file: str
    find: str
    replace: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaultSpec:
        return cls(
            fault_id=str(data["fault_id"]),
            family=str(data["family"]),
            title=str(data["title"]),
            task_types=tuple(str(value) for value in data["task_types"]),
            target_file=str(data["target_file"]),
            find=str(data["find"]),
            replace=str(data["replace"]),
        )


@dataclass(frozen=True)
class RecoveryEpisodeScore:
    episode_index: int
    fault_id: str
    fault_family: str
    diagnosis_labels: tuple[str, ...]
    diagnosis_top1_correct: bool
    diagnosis_top3_correct: bool
    pre_fault_test_metric: float
    faulted_test_metric: float
    final_test_metric: float
    normalized_fault_severity: float
    normalized_recovery_gain: float
    recovery_fraction: float | None
    fully_recovered: bool
    catastrophic_forgetting: bool
    recovery_compute_units: float | None
    budget_compliant: bool
    guardrails_passed: bool
    human_interventions: int


@dataclass(frozen=True)
class RecoveryScorecard:
    schema_version: str
    protocol: str
    run_id: str
    task_id: str
    agent_id: str
    random_seed: int
    n_episodes: int
    n_scorable_episodes: int
    mean_recovery_fraction: float | None
    full_recovery_rate: float | None
    autonomous_recovery_rate: float | None
    mean_recovery_compute_units: float | None
    diagnosis_top1_accuracy: float
    diagnosis_top3_accuracy: float
    catastrophic_forgetting_rate: float | None
    cumulative_unrecovered_fraction: float | None
    final_clean_retention: float | None
    autonomous_run: bool
    self_evolution_recovery_verified: bool
    episodes: tuple[RecoveryEpisodeScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_fault_catalog() -> dict[str, FaultSpec]:
    payload = read_json(resolve_asset(FAULT_CATALOG_PATH))
    faults = [FaultSpec.from_dict(value) for value in payload.get("faults", [])]
    if not faults or len({fault.fault_id for fault in faults}) != len(faults):
        raise RecoveryError("recovery fault catalog is empty or contains duplicate IDs")
    return {fault.fault_id: fault for fault in faults}


def task_faults(manifest: TaskManifest) -> tuple[FaultSpec, ...]:
    protocol = manifest.recovery_protocol
    if not protocol.get("enabled", False):
        raise RecoveryError(f"task {manifest.task_id} does not enable recovery evaluation")
    catalog = load_fault_catalog()
    requested = tuple(str(value) for value in protocol.get("fault_catalog", ()))
    missing = sorted(set(requested) - set(catalog))
    if missing:
        raise RecoveryError("unknown recovery faults: " + ", ".join(missing))
    faults = tuple(catalog[fault_id] for fault_id in requested)
    incompatible = [
        fault.fault_id for fault in faults if manifest.task_type not in fault.task_types
    ]
    if incompatible:
        raise RecoveryError("faults incompatible with task type: " + ", ".join(incompatible))
    return faults


def sample_faults(
    manifest: TaskManifest,
    *,
    random_seed: int,
    episodes: int | None = None,
) -> tuple[FaultSpec, ...]:
    available = list(task_faults(manifest))
    count = int(
        manifest.recovery_protocol.get("default_episodes", 2) if episodes is None else episodes
    )
    if count < 2:
        raise RecoveryError("recovery evaluation requires at least two episodes")
    if count > len(available):
        raise RecoveryError(
            f"requested {count} episodes but task exposes only {len(available)} unique faults"
        )
    ordered = sorted(
        available,
        key=lambda fault: sha256(f"{random_seed}:{fault.fault_id}".encode()).digest(),
    )
    return tuple(ordered[:count])


def _copy_workspace(source: Path, target: Path) -> None:
    if target.exists():
        raise RecoveryError(f"workspace target already exists: {target}")

    def ignore(_: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"__pycache__", ".pytest_cache", "artifacts", "predictions"}
        }

    shutil.copytree(source, target, ignore=ignore)


def inject_fault(source: str | Path, target: str | Path, fault: FaultSpec) -> str:
    """Copy a workspace and deterministically inject exactly one catalogued fault."""
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    if not source_path.is_dir():
        raise RecoveryError(f"fault source workspace is missing: {source_path}")
    _copy_workspace(source_path, target_path)
    file_path = (target_path / fault.target_file).resolve()
    try:
        file_path.relative_to(target_path)
    except ValueError as exc:
        raise RecoveryError(f"fault target escapes workspace: {fault.target_file}") from exc
    if not file_path.is_file():
        raise RecoveryError(f"fault target file is missing: {fault.target_file}")
    content = file_path.read_text(encoding="utf-8")
    occurrences = content.count(fault.find)
    if occurrences != 1:
        raise RecoveryError(
            f"fault {fault.fault_id} expected one source match, found {occurrences}"
        )
    file_path.write_text(content.replace(fault.find, fault.replace, 1), encoding="utf-8")
    return directory_digest(target_path)


def record_recovery_diagnosis(run_dir: str | Path, labels: list[str]) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    context_path = root / "recovery-context.json"
    if not context_path.is_file():
        raise RecoveryError("diagnosis target is not a recovery episode")
    if (root / "public-run.json").exists():
        raise RecoveryError("episode is already finalized")
    target = root / "diagnosis.json"
    if target.exists():
        raise RecoveryError("diagnosis was already submitted for this episode")
    cleaned = [str(label).strip() for label in labels if str(label).strip()]
    if not cleaned or len(cleaned) > 3 or len(set(cleaned)) != len(cleaned):
        raise RecoveryError("submit one to three unique diagnosis labels")
    context = read_json(context_path)
    allowed = {str(label) for label in context.get("fault_label_vocabulary", [])}
    known = set(load_fault_catalog())
    if not allowed or not allowed <= known:
        raise RecoveryError("recovery episode has an invalid fault-label vocabulary")
    if any(label not in allowed for label in cleaned):
        raise RecoveryError("diagnosis contains a fault label unavailable for this task")
    payload = {"labels": cleaned, "submitted_unix_seconds": time.time()}
    write_json(target, payload)
    return payload


def _run_candidate_public(
    manifest: TaskManifest,
    *,
    task_path: Path,
    prepared: Path,
    workspace: Path,
    output: Path,
    timeout_seconds: float,
    random_seed: int,
    allocated_gpus: int,
    sandbox_image: str | None,
) -> float:
    options = {
        "prepared_dir": prepared,
        "workspace": workspace,
        "split_name": feedback_split(manifest),
        "output_dir": output,
        "timeout_seconds": timeout_seconds,
        "random_seed": random_seed,
        "allocated_gpus": allocated_gpus,
    }
    if sandbox_image:
        result = execute_candidate_docker(
            manifest,
            task_path=task_path,
            image=sandbox_image,
            trusted_data=False,
            **options,
        )
    else:
        result = execute_candidate(manifest, **options)
    if result.status != "valid":
        raise RecoveryError(f"pre-fault candidate failed public evaluation: {result.error}")
    return float(result.metrics[manifest.primary_metric.name])


def _episode_environment(
    *,
    episode_root: Path,
    outer_root: Path,
    task: Path,
    prepared: Path,
    manifest: TaskManifest,
    agent_id: str,
    random_seed: int,
    episode_index: int,
    episode_count: int,
    target_public_metric: float,
) -> dict[str, str]:
    labels = [fault.fault_id for fault in task_faults(manifest)]
    return {
        "RECEVOBENCH_RUN_DIR": str(episode_root),
        "RECEVOBENCH_WORKSPACE": str(episode_root / "workspace"),
        "RECEVOBENCH_PUBLIC_DATA": str(prepared / "public"),
        "RECEVOBENCH_TASK_PATH": str(task),
        "RECEVOBENCH_FEEDBACK_SPLIT": feedback_split(manifest),
        "RECEVOBENCH_RANDOM_SEED": str(random_seed),
        "RECEVOBENCH_AGENT_ID": agent_id,
        "RECEVOBENCH_RECOVERY_EPISODE": str(episode_index),
        "RECEVOBENCH_RECOVERY_EPISODES": str(episode_count),
        "RECEVOBENCH_RECOVERY_TARGET_PUBLIC_METRIC": repr(target_public_metric),
        "RECEVOBENCH_RECOVERY_FAULT_LABELS": json.dumps(labels),
        "RECEVOBENCH_AGENT_MEMORY": str(outer_root / "agent-memory"),
        "RECEVOBENCH_TRIAL_COMMAND_JSON": json.dumps(
            [os.sys.executable, "-m", "recevobench.cli", "trial", "--run-dir", str(episode_root)]
        ),
        "RECEVOBENCH_DIAGNOSE_COMMAND_JSON": json.dumps(
            [
                os.sys.executable,
                "-m",
                "recevobench.cli",
                "recovery-diagnose",
                "--run-dir",
                str(episode_root),
            ]
        ),
    }


def _run_local_episode(
    *,
    episode_root: Path,
    command: list[str],
    environment: dict[str, str],
    maximum_wall: float,
) -> None:
    logs = episode_root / "agent-logs"
    logs.mkdir()
    timed_out = False
    with (logs / "stdout.log").open("wb") as stdout, (logs / "stderr.log").open("wb") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=episode_root / "workspace",
                env={**os.environ, **environment},
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
        episode_root / "agent-exit.json",
        {"exit_code": exit_code, "timed_out": timed_out, "command": command},
    )


def _run_docker_episode(
    *,
    episode_root: Path,
    outer_root: Path,
    task: Path,
    prepared: Path,
    command: list[str],
    environment: dict[str, str],
    image: str,
    maximum_wall: float,
    allocated_gpus: int,
    network: str,
    memory: str,
    cpus: float,
) -> None:
    from .control import ControlServer

    with tempfile.TemporaryDirectory(prefix="recevobench-recovery-control-") as temporary:
        control_directory = Path(temporary)
        socket_path = control_directory / "control.sock"
        container_environment = {
            key: value
            for key, value in environment.items()
            if key.startswith("RECEVOBENCH_RECOVERY_")
        }
        container_environment.update(
            {
                "RECEVOBENCH_AGENT_ID": environment["RECEVOBENCH_AGENT_ID"],
                "RECEVOBENCH_FEEDBACK_SPLIT": environment["RECEVOBENCH_FEEDBACK_SPLIT"],
                "RECEVOBENCH_RANDOM_SEED": environment["RECEVOBENCH_RANDOM_SEED"],
                "RECEVOBENCH_TRIAL_COMMAND_JSON": json.dumps(
                    [
                        "recevobench",
                        "remote-trial",
                        "--socket",
                        "/benchmark/control/control.sock",
                    ]
                ),
                "RECEVOBENCH_DIAGNOSE_COMMAND_JSON": json.dumps(
                    [
                        "recevobench",
                        "remote-recovery-diagnose",
                        "--socket",
                        "/benchmark/control/control.sock",
                    ]
                ),
            }
        )
        container_environment["RECEVOBENCH_AGENT_MEMORY"] = "/benchmark/memory"
        docker_command = build_agent_docker_command(
            image=image,
            workspace=episode_root / "workspace",
            public_data=prepared / "public",
            task_path=task,
            control_directory=control_directory,
            command=command,
            allocated_gpus=allocated_gpus,
            network=network,
            memory=memory,
            cpus=cpus,
            extra_environment=container_environment,
            extra_mounts=((outer_root / "agent-memory", "/benchmark/memory", False),),
        )
        logs = episode_root / "agent-logs"
        logs.mkdir()
        timed_out = False
        with (
            ControlServer(socket_path, episode_root),
            (logs / "stdout.log").open("wb") as stdout,
            (logs / "stderr.log").open("wb") as stderr,
        ):
            try:
                completed = subprocess.run(
                    docker_command,
                    check=False,
                    timeout=maximum_wall,
                    stdout=stdout,
                    stderr=stderr,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
                timed_out = True
        write_json(
            episode_root / "agent-exit.json",
            {
                "exit_code": exit_code,
                "timed_out": timed_out,
                "container_image": image,
                "network": network,
                "command": command,
            },
        )


def run_recovery_agent(
    *,
    task_path: str | Path,
    prepared_dir: str | Path,
    run_dir: str | Path,
    agent_id: str,
    random_seed: int,
    command: list[str],
    episodes: int | None = None,
    timeout_seconds: float = 3600.0,
    allocated_gpus: int = 1,
    image: str | None = None,
    network: str = "none",
    memory: str = "32g",
    cpus: float = 8.0,
) -> dict[str, Any]:
    """Run one agent process per hidden-fault episode with persistent public memory."""
    if not command:
        raise RecoveryError("recovery runner requires an agent command")
    if image and not docker_available():
        raise RecoveryError("Docker is required when a recovery agent image is supplied")
    task = resolve_task_path(task_path)
    prepared = Path(prepared_dir).resolve()
    root = Path(run_dir).resolve()
    if root.exists():
        raise RecoveryError(f"recovery run directory already exists: {root}")
    manifest = load_manifest(task)
    verify_dataset_card(prepared)
    faults = sample_faults(manifest, random_seed=random_seed, episodes=episodes)
    root.mkdir(parents=True)
    private = root / "private"
    private.mkdir()
    (root / "agent-memory").mkdir()
    initial = private / "initial-workspace"
    copy_seed_workspace(manifest, initial)
    catalog_path = resolve_asset(FAULT_CATALOG_PATH)
    lock = {
        "schema_version": "1.0",
        "protocol": "hidden-fault-multi-episode-v1",
        "run_id": f"{manifest.task_id}-{agent_id}-recovery-{secrets.token_hex(6)}",
        "task_id": manifest.task_id,
        "agent_id": agent_id,
        "random_seed": random_seed,
        "task_path": str(task),
        "task_sha256": sha256_file(task),
        "prepared_dir": str(prepared),
        "dataset_sha256": directory_digest(prepared, ignore=()),
        "fault_catalog_path": str(catalog_path),
        "fault_catalog_sha256": sha256_file(catalog_path),
        "fault_schedule": [fault.fault_id for fault in faults],
        "episode_count": len(faults),
        "initial_workspace_sha256": directory_digest(initial),
        "created_unix_seconds": time.time(),
        "runner": "docker" if image else "local-trusted-development",
        "image": image,
    }
    write_json(private / "recovery-lock.json", lock)
    public_protocol = {
        "schema_version": "1.0",
        "protocol": lock["protocol"],
        "run_id": lock["run_id"],
        "task_id": manifest.task_id,
        "episode_count": len(faults),
        "fault_label_vocabulary": [fault.fault_id for fault in task_faults(manifest)],
        "selection": "seeded_without_replacement",
    }
    write_json(root / "recovery-public.json", public_protocol)

    current = initial
    for index, fault in enumerate(faults, start=1):
        # The public evaluation seed must not reveal the private fault-schedule seed.
        episode_seed = 1_000_000 + index
        private_episode = private / "episodes" / f"{index:03d}"
        pre_fault = private_episode / "pre-fault-workspace"
        _copy_workspace(current, pre_fault)
        pre_fault_sha = directory_digest(pre_fault)
        target_public = _run_candidate_public(
            manifest,
            task_path=task,
            prepared=prepared,
            workspace=pre_fault,
            output=private_episode / "pre-fault-public",
            timeout_seconds=timeout_seconds,
            random_seed=episode_seed,
            allocated_gpus=allocated_gpus,
            sandbox_image=image,
        )
        faulted = private_episode / "faulted-build"
        injected_sha = inject_fault(pre_fault, faulted, fault)
        write_json(
            private_episode / "fault.json",
            {
                "fault_id": fault.fault_id,
                "fault_family": fault.family,
                "pre_fault_workspace_sha256": pre_fault_sha,
                "faulted_workspace_sha256": injected_sha,
            },
        )
        episode_root = root / "episodes" / f"{index:03d}"
        episode_lock = initialize_campaign(
            task_path=task,
            prepared_dir=prepared,
            run_dir=episode_root,
            agent_id=agent_id,
            random_seed=episode_seed,
            timeout_seconds=timeout_seconds,
            allocated_gpus=allocated_gpus,
            sandbox_image=image,
            seed_workspace_source=faulted,
            run_id=f"{lock['run_id']}-episode-{index:03d}",
        )
        faulted_public = float(episode_lock["baseline_public_metric"])
        context = {
            "episode_index": index,
            "episode_count": len(faults),
            "target_public_metric": target_public,
            "faulted_public_metric": faulted_public,
            "fault_label_vocabulary": public_protocol["fault_label_vocabulary"],
        }
        write_json(episode_root / "recovery-context.json", context)
        environment = _episode_environment(
            episode_root=episode_root,
            outer_root=root,
            task=task,
            prepared=prepared,
            manifest=manifest,
            agent_id=agent_id,
            random_seed=episode_seed,
            episode_index=index,
            episode_count=len(faults),
            target_public_metric=target_public,
        )
        maximum_wall = manifest.budget.max_wall_seconds or timeout_seconds
        if image:
            _run_docker_episode(
                episode_root=episode_root,
                outer_root=root,
                task=task,
                prepared=prepared,
                command=command,
                environment=environment,
                image=image,
                maximum_wall=maximum_wall,
                allocated_gpus=allocated_gpus,
                network=network,
                memory=memory,
                cpus=cpus,
            )
        else:
            _run_local_episode(
                episode_root=episode_root,
                command=command,
                environment=environment,
                maximum_wall=maximum_wall,
            )
        final_path = episode_root / "final-candidate.txt"
        final_id = final_path.read_text(encoding="utf-8").strip() if final_path.is_file() else None
        run = finalize_campaign(run_dir=episode_root, final_candidate_id=final_id)
        selected = (
            episode_root / "seed-workspace"
            if run.final_candidate_id == "seed"
            else episode_root / "candidates" / run.final_candidate_id
        )
        carried = private_episode / "selected-workspace"
        _copy_workspace(selected, carried)
        write_json(
            private_episode / "selection.json",
            {
                "final_candidate_id": run.final_candidate_id,
                "selected_workspace_sha256": directory_digest(carried),
            },
        )
        current = carried
    write_json(root / "recovery-run-complete.json", {**public_protocol, "complete": True})
    return public_protocol


def _recovery_compute_units(
    metric: MetricSpec,
    bundle: EvaluationBundle,
    target: float,
) -> float | None:
    consumed = 0.0
    threshold = metric.improvement_threshold(target)
    for trial in bundle.run.trials:
        consumed += (
            trial.compute_seconds if trial.compute_seconds is not None else trial.gpu_seconds
        )
        if (
            trial.status == "valid"
            and trial.hidden_select_metric is not None
            and metric.orient(float(trial.hidden_select_metric))
            >= metric.orient(target) - threshold
        ):
            return consumed / bundle.run.baseline_compute_seconds
    return None


def grade_recovery_campaign(run_dir: str | Path) -> RecoveryScorecard:
    """Trusted grading that reveals faults only after every episode is frozen."""
    root = Path(run_dir).resolve()
    lock = read_json(root / "private" / "recovery-lock.json")
    if not (root / "recovery-run-complete.json").is_file():
        raise RecoveryError("recovery run is not complete")
    task = resolve_task_path(lock["task_path"])
    manifest = load_manifest(task)
    prepared = Path(str(lock["prepared_dir"]))
    verify_dataset_card(prepared)
    if sha256_file(task) != lock.get("task_sha256"):
        raise RecoveryError("recovery task changed after initialization")
    if directory_digest(prepared, ignore=()) != lock.get("dataset_sha256"):
        raise RecoveryError("recovery dataset changed after initialization")
    catalog_path = resolve_asset(FAULT_CATALOG_PATH)
    if sha256_file(catalog_path) != lock.get("fault_catalog_sha256"):
        raise RecoveryError("recovery fault catalog changed after initialization")
    initial_workspace = root / "private" / "initial-workspace"
    if directory_digest(initial_workspace) != lock.get("initial_workspace_sha256"):
        raise RecoveryError("recovery initial workspace changed after initialization")
    catalog = load_fault_catalog()
    metric = manifest.primary_metric
    scale = metric.scale or 1.0
    scores: list[RecoveryEpisodeScore] = []
    expected_pre_fault_sha = str(lock["initial_workspace_sha256"])

    for index, fault_id in enumerate(lock["fault_schedule"], start=1):
        fault = catalog[str(fault_id)]
        episode_root = root / "episodes" / f"{index:03d}"
        private_episode = root / "private" / "episodes" / f"{index:03d}"
        fault_record = read_json(private_episode / "fault.json")
        if fault_record.get("fault_id") != fault.fault_id:
            raise RecoveryError(f"recovery fault record mismatch in episode {index}")
        actual_pre_fault_sha = directory_digest(private_episode / "pre-fault-workspace")
        if (
            actual_pre_fault_sha != expected_pre_fault_sha
            or fault_record.get("pre_fault_workspace_sha256") != actual_pre_fault_sha
        ):
            raise RecoveryError(f"recovery state chain mismatch in episode {index}")
        actual_faulted_sha = directory_digest(private_episode / "faulted-build")
        if fault_record.get("faulted_workspace_sha256") != actual_faulted_sha:
            raise RecoveryError(f"faulted workspace changed in episode {index}")
        bundle, card = grade_campaign(episode_root)
        episode_lock, _, _ = _load_campaign(episode_root)
        pre_fault_hidden = _execute_campaign_candidate(
            episode_lock,
            manifest,
            workspace=private_episode / "pre-fault-workspace",
            split_name=_hidden_split(manifest),
            output_dir=private_episode / "pre-fault-hidden",
            trusted_data=True,
        )
        pre_fault_test = _execute_campaign_candidate(
            episode_lock,
            manifest,
            workspace=private_episode / "pre-fault-workspace",
            split_name=_sealed_split(manifest),
            output_dir=private_episode / "pre-fault-test",
            trusted_data=True,
        )
        if pre_fault_hidden.status != "valid" or pre_fault_test.status != "valid":
            raise RecoveryError(f"pre-fault evaluation failed in episode {index}")
        name = metric.name
        pre_hidden_value = float(pre_fault_hidden.metrics[name])
        pre_test_value = float(pre_fault_test.metrics[name])
        faulted_test_value = float(bundle.baseline_test_metrics[name])
        final_test_value = float(bundle.final_test_metrics[name])
        severity_raw = metric.orient(pre_test_value) - metric.orient(faulted_test_value)
        recovery_raw = metric.orient(final_test_value) - metric.orient(faulted_test_value)
        scorable = severity_raw > metric.improvement_threshold(pre_test_value)
        recovery_fraction = min(1.0, max(0.0, recovery_raw / severity_raw)) if scorable else None
        final_deficit = metric.orient(pre_test_value) - metric.orient(final_test_value)
        forgetting = final_deficit > metric.improvement_threshold(pre_test_value)
        recovery_guardrails = tuple(
            _guardrail_result(
                spec,
                pre_fault_test.metrics,
                bundle.final_guardrail_metrics,
            )
            for spec in manifest.guardrails
        )
        recovery_guardrails_passed = all(result.passed for result in recovery_guardrails)
        fully_recovered = (
            scorable and not forgetting and card.budget_compliant and recovery_guardrails_passed
        )
        diagnosis = (
            read_json(episode_root / "diagnosis.json")
            if (episode_root / "diagnosis.json").is_file()
            else {"labels": []}
        )
        labels = tuple(str(value) for value in diagnosis.get("labels", []))
        scores.append(
            RecoveryEpisodeScore(
                episode_index=index,
                fault_id=fault.fault_id,
                fault_family=fault.family,
                diagnosis_labels=labels,
                diagnosis_top1_correct=bool(labels) and labels[0] == fault.fault_id,
                diagnosis_top3_correct=fault.fault_id in labels[:3],
                pre_fault_test_metric=pre_test_value,
                faulted_test_metric=faulted_test_value,
                final_test_metric=final_test_value,
                normalized_fault_severity=max(0.0, severity_raw / scale),
                normalized_recovery_gain=recovery_raw / scale,
                recovery_fraction=recovery_fraction,
                fully_recovered=fully_recovered,
                catastrophic_forgetting=forgetting,
                recovery_compute_units=_recovery_compute_units(
                    metric,
                    bundle,
                    pre_hidden_value,
                ),
                budget_compliant=card.budget_compliant,
                guardrails_passed=recovery_guardrails_passed,
                human_interventions=card.human_interventions,
            )
        )
        selection = read_json(private_episode / "selection.json")
        selected_sha = directory_digest(private_episode / "selected-workspace")
        if selection.get("selected_workspace_sha256") != selected_sha:
            raise RecoveryError(f"selected workspace changed in episode {index}")
        if selection.get("final_candidate_id") != bundle.run.final_candidate_id:
            raise RecoveryError(f"selected candidate mismatch in episode {index}")
        expected_pre_fault_sha = selected_sha

    scorable_scores = [score for score in scores if score.recovery_fraction is not None]
    recovery_fractions = [float(score.recovery_fraction) for score in scorable_scores]
    recovery_times = [
        float(score.recovery_compute_units)
        for score in scorable_scores
        if score.recovery_compute_units is not None
    ]
    autonomous = all(score.human_interventions == 0 for score in scores)
    initial = scores[0].pre_fault_test_metric if scores else None
    final = scores[-1].final_test_metric if scores else None
    final_retention = (
        (metric.orient(final) - metric.orient(initial)) / scale
        if initial is not None and final is not None
        else None
    )
    initial_tolerance = (
        metric.improvement_threshold(initial) / scale if initial is not None else 0.0
    )
    verified = (
        len(scores) >= 2
        and len(scorable_scores) == len(scores)
        and all(score.fully_recovered for score in scores)
        and autonomous
        and final_retention is not None
        and final_retention >= -initial_tolerance
    )
    scorecard = RecoveryScorecard(
        schema_version="1.0",
        protocol="hidden-fault-multi-episode-v1",
        run_id=str(lock["run_id"]),
        task_id=manifest.task_id,
        agent_id=str(lock["agent_id"]),
        random_seed=int(lock["random_seed"]),
        n_episodes=len(scores),
        n_scorable_episodes=len(scorable_scores),
        mean_recovery_fraction=(fmean(recovery_fractions) if recovery_fractions else None),
        full_recovery_rate=(
            fmean(float(score.fully_recovered) for score in scorable_scores)
            if scorable_scores
            else None
        ),
        autonomous_recovery_rate=(
            fmean(
                float(score.fully_recovered and score.human_interventions == 0)
                for score in scorable_scores
            )
            if scorable_scores
            else None
        ),
        mean_recovery_compute_units=(fmean(recovery_times) if recovery_times else None),
        diagnosis_top1_accuracy=fmean(float(score.diagnosis_top1_correct) for score in scores),
        diagnosis_top3_accuracy=fmean(float(score.diagnosis_top3_correct) for score in scores),
        catastrophic_forgetting_rate=(
            fmean(float(score.catastrophic_forgetting) for score in scorable_scores)
            if scorable_scores
            else None
        ),
        cumulative_unrecovered_fraction=(
            fmean(1.0 - value for value in recovery_fractions) if recovery_fractions else None
        ),
        final_clean_retention=final_retention,
        autonomous_run=autonomous,
        self_evolution_recovery_verified=verified,
        episodes=tuple(scores),
    )
    write_json(root / "recovery-scorecard.json", scorecard)
    return scorecard
