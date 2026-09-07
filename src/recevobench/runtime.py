"""Canonical candidate execution with deterministic commands and audit logs."""

from __future__ import annotations

import csv
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import directory_digest, total_file_bytes
from .assets import resolve_asset
from .contracts import TaskManifest
from .evaluation import evaluate_prepared_split
from .io import write_json


class RuntimeError(ValueError):
    """Raised when a candidate cannot be executed under the task contract."""


@dataclass(frozen=True)
class PhaseResult:
    command: tuple[str, ...]
    exit_code: int
    wall_seconds: float
    cpu_seconds: float
    stdout_path: str
    stderr_path: str


@dataclass(frozen=True)
class CandidateResult:
    status: str
    metrics: dict[str, float]
    train: PhaseResult | None
    predict: PhaseResult | None
    wall_seconds: float
    cpu_seconds: float
    gpu_seconds: float
    compute_seconds: float
    candidate_sha256: str
    model_sha256: str | None
    prediction_sha256: str
    error: str = ""
    latency_samples_ms: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateResult:
        converted = dict(data)
        for name in ("train", "predict"):
            phase = converted.get(name)
            if phase is not None:
                phase = dict(phase)
                phase["command"] = tuple(phase["command"])
                converted[name] = PhaseResult(**phase)
        converted["latency_samples_ms"] = tuple(
            float(value) for value in converted.get("latency_samples_ms", ())
        )
        return cls(**converted)


def _find_split(prepared: Path, name: str) -> Path:
    for visibility in ("public", "private"):
        candidate = prepared / visibility / f"{name}.csv"
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"prepared split not found: {name}")


def training_split(manifest: TaskManifest) -> str:
    for split in manifest.splits:
        if split.visibility == "public" and "train" in split.name:
            return split.name
    raise RuntimeError("task has no public training split")


def feedback_split(manifest: TaskManifest) -> str:
    for split in manifest.splits:
        if split.visibility == "public" and "feedback" in split.name:
            return split.name
    raise RuntimeError("task has no public feedback split")


def _format_command(template: Sequence[str], values: Mapping[str, str]) -> list[str]:
    command = [part.format_map(values) for part in template]
    if command and command[0] in {"python", "python3"}:
        command[0] = sys.executable
    return command


def _run_phase(
    command: Sequence[str],
    *,
    cwd: Path,
    log_dir: Path,
    name: str,
    timeout: float,
    env: Mapping[str, str],
) -> PhaseResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env={**os.environ, **env},
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    elapsed = time.monotonic() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (
        usage_after.ru_utime + usage_after.ru_stime - usage_before.ru_utime - usage_before.ru_stime
    )
    return PhaseResult(
        command=tuple(command),
        exit_code=exit_code,
        wall_seconds=elapsed,
        cpu_seconds=max(0.0, cpu),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _prediction_digest(path: Path) -> str:
    return directory_digest(path, ignore=())


def _validate_candidate(manifest: TaskManifest, workspace: Path) -> None:
    required = manifest.artifact_policy.get("required_candidate_files", [])
    missing = [name for name in required if not (workspace / str(name)).is_file()]
    if missing:
        raise RuntimeError("candidate misses required files: " + ", ".join(missing))
    maximum = int(manifest.artifact_policy.get("max_candidate_bytes", 0) or 0)
    if maximum and total_file_bytes(workspace) > maximum:
        raise RuntimeError(f"candidate exceeds max_candidate_bytes={maximum}")


def _write_latency_sample(
    source: Path,
    destination: Path,
    *,
    maximum_rows: int,
    maximum_users: int | None,
) -> int:
    """Create a deterministic evaluator-owned inference micro-batch."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    users: set[str] = set()
    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise RuntimeError("latency source has no CSV header")
        with destination.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                user = str(row.get("user_id", ""))
                if maximum_users is not None and user not in users:
                    if len(users) >= maximum_users:
                        continue
                    users.add(user)
                writer.writerow(row)
                written += 1
                if written >= maximum_rows:
                    break
    if written == 0:
        raise RuntimeError("latency source contains no rows")
    return written


def _latency_probe(
    manifest: TaskManifest,
    *,
    candidate: Path,
    split: Path,
    values: Mapping[str, str],
    output: Path,
    logs: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[tuple[float, ...], float, float, str]:
    runtime = manifest.runtime
    repeats = max(1, int(runtime.get("latency_repeats", 5)))
    maximum_rows = max(1, int(runtime.get("latency_sample_rows", 128)))
    ranking = str(runtime.get("prediction_format", "")).startswith("ranking")
    maximum_users = max(1, int(runtime.get("latency_sample_users", 1))) if ranking else None
    sample = output / "latency" / "input.csv"
    _write_latency_sample(
        split,
        sample,
        maximum_rows=maximum_rows,
        maximum_users=maximum_users,
    )
    wall_total = 0.0
    cpu_total = 0.0
    samples: list[float] = []
    # One warm-up is excluded from the percentile but included in consumed compute.
    for index in range(repeats + 1):
        remaining = timeout_seconds - wall_total
        if remaining <= 0:
            return (), wall_total, cpu_total, "latency probe exhausted timeout"
        probe_predictions = output / "latency" / f"predictions-{index:02d}"
        probe_predictions.mkdir(parents=True)
        probe_values = {
            **values,
            "split": str(sample),
            "predictions": str(probe_predictions),
        }
        phase = _run_phase(
            _format_command(runtime["predict_command"], probe_values),
            cwd=candidate,
            log_dir=logs,
            name=f"latency-{index:02d}",
            timeout=max(0.001, remaining),
            env=env,
        )
        wall_total += phase.wall_seconds
        cpu_total += phase.cpu_seconds
        if phase.exit_code != 0:
            return (), wall_total, cpu_total, f"latency probe exited with {phase.exit_code}"
        if index:
            samples.append(phase.wall_seconds * 1000.0)
    return tuple(samples), wall_total, cpu_total, ""


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def execute_candidate(
    manifest: TaskManifest,
    *,
    prepared_dir: str | Path,
    workspace: str | Path,
    split_name: str,
    output_dir: str | Path,
    timeout_seconds: float,
    random_seed: int,
    allocated_gpus: int = 1,
) -> CandidateResult:
    """Train, predict and grade a candidate in a benchmark-controlled directory."""
    prepared = Path(prepared_dir).resolve()
    candidate = Path(workspace).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions = output / "predictions"
    predictions.mkdir()
    logs = output / "logs"
    _validate_candidate(manifest, candidate)
    runtime = manifest.runtime
    if not runtime.get("train_command") or not runtime.get("predict_command"):
        raise RuntimeError("task runtime must define train_command and predict_command")
    model = output / str(runtime.get("model_artifact", "artifacts/model.json"))
    model.parent.mkdir(parents=True, exist_ok=True)
    split_path = _find_split(prepared, split_name)
    values = {
        "train": str(_find_split(prepared, training_split(manifest))),
        "split": str(split_path),
        "catalog": str(prepared / "public" / "catalog.csv"),
        "model": str(model),
        "predictions": str(predictions),
        "workspace": str(candidate),
        "seed": str(random_seed),
    }
    env = {
        "PYTHONHASHSEED": str(random_seed),
        "RECEVOBENCH_RANDOM_SEED": str(random_seed),
        "RECEVOBENCH_TASK_ID": manifest.task_id,
    }
    train_command = _format_command(runtime["train_command"], values)
    train = _run_phase(
        train_command,
        cwd=candidate,
        log_dir=logs,
        name="train",
        timeout=timeout_seconds,
        env=env,
    )
    if train.exit_code != 0:
        status = "timeout" if train.exit_code == 124 else "crash"
        result = CandidateResult(
            status=status,
            metrics={},
            train=train,
            predict=None,
            wall_seconds=train.wall_seconds,
            cpu_seconds=train.cpu_seconds,
            gpu_seconds=train.wall_seconds * max(0, allocated_gpus),
            compute_seconds=train.wall_seconds * max(1, allocated_gpus),
            candidate_sha256=directory_digest(candidate),
            model_sha256=None,
            prediction_sha256=directory_digest(predictions, ignore=()),
            error=f"training exited with {train.exit_code}",
        )
        write_json(output / "candidate-result.json", result)
        return result

    predict_command = _format_command(runtime["predict_command"], values)
    remaining = max(0.001, timeout_seconds - train.wall_seconds)
    predict = _run_phase(
        predict_command,
        cwd=candidate,
        log_dir=logs,
        name="predict",
        timeout=remaining,
        env=env,
    )
    wall = train.wall_seconds + predict.wall_seconds
    cpu = train.cpu_seconds + predict.cpu_seconds
    if predict.exit_code != 0:
        status = "timeout" if predict.exit_code == 124 else "crash"
        result = CandidateResult(
            status=status,
            metrics={},
            train=train,
            predict=predict,
            wall_seconds=wall,
            cpu_seconds=cpu,
            gpu_seconds=wall * max(0, allocated_gpus),
            compute_seconds=wall * max(1, allocated_gpus),
            candidate_sha256=directory_digest(candidate),
            model_sha256=(directory_digest(model.parent, ignore=()) if model.exists() else None),
            prediction_sha256=_prediction_digest(predictions),
            error=f"prediction exited with {predict.exit_code}",
        )
        write_json(output / "candidate-result.json", result)
        return result

    latency_samples, latency_wall, latency_cpu, latency_error = _latency_probe(
        manifest,
        candidate=candidate,
        split=split_path,
        values=values,
        output=output,
        logs=logs,
        env=env,
        timeout_seconds=max(0.001, timeout_seconds - wall),
    )
    wall += latency_wall
    cpu += latency_cpu
    try:
        if latency_error:
            raise RuntimeError(latency_error)
        metrics = evaluate_prepared_split(manifest, prepared, split_name, predictions)
        status = "valid"
        error = ""
    except Exception as exc:
        metrics = {}
        status = "invalid"
        error = str(exc)
    if latency_samples:
        metrics["InferenceP95Ms"] = _nearest_rank_percentile(latency_samples, 0.95)
    result = CandidateResult(
        status=status,
        metrics=metrics,
        train=train,
        predict=predict,
        wall_seconds=wall,
        cpu_seconds=cpu,
        gpu_seconds=wall * max(0, allocated_gpus),
        compute_seconds=wall * max(1, allocated_gpus),
        candidate_sha256=directory_digest(candidate),
        model_sha256=(directory_digest(model.parent, ignore=()) if model.exists() else None),
        prediction_sha256=_prediction_digest(predictions),
        error=error,
        latency_samples_ms=latency_samples,
    )
    write_json(output / "candidate-result.json", result)
    write_json(output / "environment.json", environment_fingerprint())
    return result


def environment_fingerprint() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def copy_seed_workspace(manifest: TaskManifest, target: Path) -> None:
    source = resolve_asset(str(manifest.runtime.get("seed_workspace", "")))
    if not source.is_dir():
        raise RuntimeError(f"seed workspace does not exist: {source}")
    if target.exists():
        raise RuntimeError(f"workspace target already exists: {target}")
    shutil.copytree(source, target)
