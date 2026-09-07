"""Docker execution for untrusted candidate code."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .contracts import TaskManifest
from .io import read_json
from .runtime import CandidateResult, RuntimeError

DEFAULT_IMAGE = "recevobench:local"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def build_agent_docker_command(
    *,
    image: str,
    workspace: Path,
    public_data: Path,
    task_path: Path,
    control_directory: Path,
    command: Sequence[str],
    allocated_gpus: int,
    network: str = "none",
    memory: str = "32g",
    cpus: float = 8.0,
) -> list[str]:
    if not command:
        raise RuntimeError("agent container command cannot be empty")
    if network not in {"none", "bridge"}:
        raise RuntimeError("agent network must be none or bridge")
    docker = [
        "docker",
        "run",
        "--rm",
        "--network",
        network,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--memory",
        memory,
        "--cpus",
        str(cpus),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=4g",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/benchmark/workspace",
        "--mount",
        f"type=bind,src={public_data.resolve()},dst=/benchmark/data,readonly",
        "--mount",
        f"type=bind,src={task_path.resolve()},dst=/benchmark/task.json,readonly",
        "--mount",
        f"type=bind,src={control_directory.resolve()},dst=/benchmark/control",
        "--workdir",
        "/benchmark/workspace",
        "--env",
        "RECEVOBENCH_WORKSPACE=/benchmark/workspace",
        "--env",
        "RECEVOBENCH_PUBLIC_DATA=/benchmark/data",
        "--env",
        "RECEVOBENCH_TASK_PATH=/benchmark/task.json",
        "--env",
        "RECEVOBENCH_CONTROL_SOCKET=/benchmark/control/control.sock",
    ]
    if allocated_gpus > 0:
        docker.extend(["--gpus", str(allocated_gpus)])
    docker.extend(["--entrypoint", command[0], image, *command[1:]])
    return docker


def build_candidate_docker_command(
    manifest: TaskManifest,
    *,
    task_path: Path,
    prepared_dir: Path,
    workspace: Path,
    output_dir: Path,
    split_name: str,
    timeout_seconds: float,
    random_seed: int,
    allocated_gpus: int,
    image: str = DEFAULT_IMAGE,
    trusted_data: bool = False,
    memory: str = "32g",
    cpus: float = 8.0,
) -> list[str]:
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--memory",
        memory,
        "--cpus",
        str(cpus),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=4g",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/benchmark/workspace,readonly",
        "--mount",
        f"type=bind,src={task_path.resolve()},dst=/benchmark/task.json,readonly",
        "--mount",
        (
            f"type=bind,src={(prepared_dir / 'public').resolve()},"
            "dst=/benchmark/prepared/public,readonly"
        ),
        "--mount",
        f"type=bind,src={output_parent.resolve()},dst=/benchmark/output",
    ]
    if trusted_data:
        command.extend(
            [
                "--mount",
                (
                    f"type=bind,src={(prepared_dir / 'private').resolve()},"
                    "dst=/benchmark/prepared/private,readonly"
                ),
            ]
        )
    if allocated_gpus > 0:
        command.extend(["--gpus", str(allocated_gpus)])
    command.extend(
        [
            image,
            "execute-candidate",
            "--task",
            "/benchmark/task.json",
            "--prepared",
            "/benchmark/prepared",
            "--workspace",
            "/benchmark/workspace",
            "--split",
            split_name,
            "--output",
            f"/benchmark/output/{output_dir.name}",
            "--timeout",
            str(timeout_seconds),
            "--seed",
            str(random_seed),
            "--gpus",
            str(allocated_gpus),
        ]
    )
    return command


def execute_candidate_docker(
    manifest: TaskManifest,
    *,
    task_path: str | Path,
    prepared_dir: str | Path,
    workspace: str | Path,
    split_name: str,
    output_dir: str | Path,
    timeout_seconds: float,
    random_seed: int,
    allocated_gpus: int = 1,
    image: str = DEFAULT_IMAGE,
    trusted_data: bool = False,
    memory: str = "32g",
    cpus: float = 8.0,
) -> CandidateResult:
    """Execute untrusted model code with no network and read-only task/data mounts."""
    if not docker_available():
        raise RuntimeError("Docker is required for sandboxed candidate execution")
    output = Path(output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"candidate output already exists: {output}")
    command = build_candidate_docker_command(
        manifest,
        task_path=Path(task_path),
        prepared_dir=Path(prepared_dir),
        workspace=Path(workspace),
        output_dir=output,
        split_name=split_name,
        timeout_seconds=timeout_seconds,
        random_seed=random_seed,
        allocated_gpus=allocated_gpus,
        image=image,
        trusted_data=trusted_data,
        memory=memory,
        cpus=cpus,
    )
    completed = subprocess.run(
        command,
        check=False,
        timeout=timeout_seconds + 30.0,
        capture_output=True,
    )
    result_path = output / "candidate-result.json"
    if not result_path.is_file():
        stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"sandbox did not produce candidate-result.json (exit={completed.returncode}): {stderr}"
        )
    return CandidateResult.from_dict(read_json(result_path))
