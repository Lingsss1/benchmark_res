import os
from pathlib import Path

from recevobench.campaign import finalize_campaign, initialize_campaign
from recevobench.control import (
    ControlServer,
    control_request,
    finalize_remote,
    submit_remote_recovery_diagnosis,
    submit_remote_trial,
)
from recevobench.data import prepare_task
from recevobench.io import load_manifest
from recevobench.sandbox import (
    build_agent_docker_command,
    build_candidate_docker_command,
)

ROOT = Path(__file__).resolve().parents[1]


def _campaign(tmp_path: Path) -> tuple[Path, Path, Path]:
    task = ROOT / "tasks" / "amazon-seqrec-v0.json"
    manifest = load_manifest(task)
    prepared = tmp_path / "prepared"
    prepare_task(
        manifest,
        [ROOT / "tests" / "fixtures" / "amazon" / "reviews.jsonl"],
        prepared,
    )
    run_dir = tmp_path / "run"
    initialize_campaign(
        task_path=task,
        prepared_dir=prepared,
        run_dir=run_dir,
        agent_id="socket-agent",
        random_seed=9,
        timeout_seconds=10,
        allocated_gpus=0,
    )
    return task, prepared, run_dir


def test_host_control_socket_owns_trial_and_final_selection(tmp_path) -> None:
    _, _, run_dir = _campaign(tmp_path)
    socket_path = Path("/tmp") / f"rvb-{os.getpid()}-{tmp_path.name}.sock"
    with ControlServer(socket_path, run_dir):
        assert control_request(socket_path, {"action": "ping"})["ok"] is True
        response = submit_remote_trial(
            socket_path,
            candidate_id="remote-001",
            parent_id="seed",
            hypothesis="Remote no-op",
            promoted=True,
        )
        assert response["result"]["payload"]["status"] == "valid"
        finalize_remote(socket_path, "remote-001")
    run = finalize_campaign(run_dir=run_dir, final_candidate_id="remote-001")
    assert run.final_candidate_id == "remote-001"


def test_candidate_docker_command_does_not_mount_private_data(tmp_path) -> None:
    task = ROOT / "tasks" / "amazon-seqrec-v0.json"
    manifest = load_manifest(task)
    prepared = tmp_path / "prepared"
    (prepared / "public").mkdir(parents=True)
    (prepared / "private").mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = build_candidate_docker_command(
        manifest,
        task_path=task,
        prepared_dir=prepared,
        workspace=workspace,
        output_dir=tmp_path / "output" / "candidate",
        split_name="feedback_dev",
        timeout_seconds=30,
        random_seed=1,
        allocated_gpus=0,
        trusted_data=False,
    )
    rendered = " ".join(command)
    assert str(prepared / "public") in rendered
    assert str(prepared / "private") not in rendered
    assert "--network none" in rendered
    assert "--read-only" in command
    assert "--cap-drop ALL" in rendered


def test_agent_docker_mounts_only_public_data_and_control_socket(tmp_path) -> None:
    for name in ("workspace", "public", "control"):
        (tmp_path / name).mkdir()
    task = ROOT / "tasks" / "amazon-seqrec-v0.json"
    command = build_agent_docker_command(
        image="recevobench:test",
        workspace=tmp_path / "workspace",
        public_data=tmp_path / "public",
        task_path=task,
        control_directory=tmp_path / "control",
        command=["python", "-m", "recevobench.conformance_agent"],
        allocated_gpus=0,
    )
    rendered = " ".join(command)
    assert "RECEVOBENCH_CONTROL_SOCKET=/benchmark/control/control.sock" in rendered
    assert "--network none" in rendered
    assert "/benchmark/prepared/private" not in rendered


def test_recovery_control_and_persistent_memory_mount(tmp_path) -> None:
    _, _, run_dir = _campaign(tmp_path)
    (run_dir / "recovery-context.json").write_text(
        '{"fault_label_vocabulary":["ranking_signal_inverted"]}',
        encoding="utf-8",
    )
    socket_path = Path("/tmp") / f"rvb-recovery-{os.getpid()}-{tmp_path.name}.sock"
    with ControlServer(socket_path, run_dir):
        response = submit_remote_recovery_diagnosis(socket_path, ["ranking_signal_inverted"])
    assert response["result"]["labels"] == ["ranking_signal_inverted"]

    for name in ("workspace", "public", "control", "memory"):
        (tmp_path / name).mkdir(exist_ok=True)
    task = ROOT / "tasks" / "amazon-seqrec-v0.json"
    command = build_agent_docker_command(
        image="recevobench:test",
        workspace=tmp_path / "workspace",
        public_data=tmp_path / "public",
        task_path=task,
        control_directory=tmp_path / "control",
        command=["python", "-m", "recevobench.recovery_conformance_agent"],
        allocated_gpus=0,
        extra_environment={"RECEVOBENCH_RECOVERY_EPISODE": "1"},
        extra_mounts=((tmp_path / "memory", "/benchmark/memory", False),),
    )
    rendered = " ".join(command)
    assert "RECEVOBENCH_RECOVERY_EPISODE=1" in rendered
    assert "dst=/benchmark/memory" in rendered
    assert "dst=/benchmark/memory,readonly" not in rendered
    assert "/benchmark/prepared/private" not in rendered
