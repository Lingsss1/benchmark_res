"""Command-line interface for validating and scoring benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregate import aggregate_scorecards
from .assets import list_builtin_tasks
from .campaign import (
    evaluate_trial,
    finalize_campaign,
    grade_campaign,
    initialize_campaign,
    run_agent_command,
    run_agent_docker,
)
from .control import finalize_remote, submit_remote_recovery_diagnosis, submit_remote_trial
from .data import prepare_task, verify_dataset_card
from .evaluation import evaluate_prepared_split
from .io import (
    load_evaluation_bundle,
    load_manifest,
    load_run_trace,
    read_json,
    to_jsonable,
    write_json,
)
from .recovery import (
    grade_recovery_campaign,
    record_recovery_diagnosis,
    run_recovery_agent,
    task_faults,
)
from .runtime import execute_candidate
from .schema import SCHEMA_FILES, validate_artifact
from .scoring import Scorecard, score_bundle
from .sealing import validate_public_run
from .smoke import run_smoke_suite


def _print_json(value: object) -> None:
    print(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))


def _cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.task)
    bundle = load_evaluation_bundle(args.evaluation)
    if bundle.run.task_id != manifest.task_id:
        raise ValueError("evaluation run does not belong to the supplied task")
    _print_json(
        {
            "ok": True,
            "task_id": manifest.task_id,
            "run_id": bundle.run.run_id,
            "trials": len(bundle.run.trials),
        }
    )
    return 0


def _cmd_validate_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.task)
    run = load_run_trace(args.run)
    if run.task_id != manifest.task_id:
        raise ValueError("public run does not belong to the supplied task")
    validate_public_run(run)
    _print_json(
        {
            "ok": True,
            "task_id": manifest.task_id,
            "run_id": run.run_id,
            "final_candidate_id": run.final_candidate_id,
            "trials": len(run.trials),
        }
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    scorecard = score_bundle(
        load_manifest(args.task),
        load_evaluation_bundle(args.evaluation),
        artifact_root=args.artifacts,
    )
    if args.output:
        write_json(args.output, scorecard)
    else:
        _print_json(scorecard)
    return 0


def _scorecard_from_dict(data: dict[str, object]) -> Scorecard:
    from .scoring import GuardrailResult

    converted = dict(data)
    converted.setdefault("compute_seconds_used", converted.get("gpu_seconds_used", 0.0))
    evolution_defaults = {
        "self_evolution_verified": False,
        "evolution_evidence_level": "none",
        "autonomous_run": False,
        "human_interventions_per_trial": 0.0,
        "validation_test_gap": None,
        "public_hidden_spearman": None,
        "productive_trial_ratio": 0.0,
        "feedback_conversion_rate": None,
        "misleading_feedback_rate": None,
        "successful_revision_ratio": None,
        "iterative_revision_ratio": 0.0,
        "post_failure_recovery_rate": None,
        "regression_recovery_rate": None,
        "wasted_compute_fraction": 0.0,
        "best_improvement_compute_fraction": None,
        "hidden_improvement_per_compute_unit": None,
        "max_lineage_depth": 0,
        "search_branching_ratio": 0.0,
        "unique_candidate_ratio": None,
        "successful_revision_count": 0,
        "successful_iterative_revision_count": 0,
        "code_exploration_reach": None,
        "code_exploration_spread": None,
        "structural_unique_candidate_ratio": None,
    }
    for key, value in evolution_defaults.items():
        converted.setdefault(key, value)
    converted["budget_violations"] = tuple(converted.get("budget_violations", []))
    converted["guardrails"] = tuple(
        GuardrailResult(**item)
        for item in converted.get("guardrails", [])  # type: ignore[arg-type]
    )
    return Scorecard(**converted)  # type: ignore[arg-type]


def _cmd_aggregate(args: argparse.Namespace) -> int:
    cards = [_scorecard_from_dict(read_json(path)) for path in args.scorecards]
    aggregate = aggregate_scorecards(cards)
    if args.output:
        write_json(args.output, aggregate)
    else:
        _print_json(aggregate)
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.task)
    card = prepare_task(
        manifest,
        args.raw,
        args.output,
        limit=args.limit,
    )
    _print_json(card)
    return 0


def _cmd_verify_data(args: argparse.Namespace) -> int:
    _print_json(verify_dataset_card(args.prepared))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    metrics = evaluate_prepared_split(
        load_manifest(args.task),
        args.prepared,
        args.split,
        args.predictions,
    )
    if args.output:
        write_json(args.output, metrics)
    else:
        _print_json(metrics)
    return 0


def _cmd_execute_candidate(args: argparse.Namespace) -> int:
    result = execute_candidate(
        load_manifest(args.task),
        prepared_dir=args.prepared,
        workspace=args.workspace,
        split_name=args.split,
        output_dir=args.output,
        timeout_seconds=args.timeout,
        random_seed=args.seed,
        allocated_gpus=args.gpus,
    )
    _print_json(result)
    return 0 if result.status == "valid" else 3


def _cmd_campaign_init(args: argparse.Namespace) -> int:
    lock = initialize_campaign(
        task_path=args.task,
        prepared_dir=args.prepared,
        run_dir=args.run_dir,
        agent_id=args.agent_id,
        random_seed=args.seed,
        timeout_seconds=args.trial_timeout,
        allocated_gpus=args.gpus,
        sandbox_image=args.sandbox_image,
    )
    _print_json(lock)
    return 0


def _cmd_trial(args: argparse.Namespace) -> int:
    event = evaluate_trial(
        run_dir=args.run_dir,
        candidate_id=args.candidate_id,
        parent_id=args.parent_id,
        hypothesis=args.hypothesis,
        promoted=args.promoted,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        human_interventions=args.human_interventions,
    )
    _print_json(event)
    return 0 if event["payload"]["status"] == "valid" else 3


def _cmd_campaign_finalize(args: argparse.Namespace) -> int:
    _print_json(finalize_campaign(run_dir=args.run_dir, final_candidate_id=args.final_candidate_id))
    return 0


def _cmd_grade_campaign(args: argparse.Namespace) -> int:
    _, scorecard = grade_campaign(args.run_dir)
    _print_json(scorecard)
    return 0


def _cmd_validate_schema(args: argparse.Namespace) -> int:
    value = read_json(args.input)
    validate_artifact(args.kind, value)
    _print_json({"ok": True, "kind": args.kind, "input": str(args.input)})
    return 0


def _cmd_task_list(args: argparse.Namespace) -> int:
    _print_json({"tasks": list(list_builtin_tasks())})
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    _print_json(run_smoke_suite(args.output))
    return 0


def _cmd_run_agent(args: argparse.Namespace) -> int:
    command = list(args.agent_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("run-agent requires a command after --")
    run = run_agent_command(
        task_path=args.task,
        prepared_dir=args.prepared,
        run_dir=args.run_dir,
        agent_id=args.agent_id,
        random_seed=args.seed,
        command=command,
        timeout_seconds=args.trial_timeout,
        allocated_gpus=args.gpus,
        sandbox_image=args.sandbox_image,
    )
    _print_json(run)
    return 0


def _cmd_remote_trial(args: argparse.Namespace) -> int:
    response = submit_remote_trial(
        args.socket,
        candidate_id=args.candidate_id,
        parent_id=args.parent_id,
        hypothesis=args.hypothesis,
        promoted=args.promoted,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        human_interventions=args.human_interventions,
    )
    _print_json(response)
    return 0 if response["result"]["payload"]["status"] == "valid" else 3


def _cmd_remote_finalize(args: argparse.Namespace) -> int:
    _print_json(finalize_remote(args.socket, args.candidate_id))
    return 0


def _cmd_run_agent_docker(args: argparse.Namespace) -> int:
    command = list(args.agent_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("run-agent-docker requires a command after --")
    _print_json(
        run_agent_docker(
            task_path=args.task,
            prepared_dir=args.prepared,
            run_dir=args.run_dir,
            agent_id=args.agent_id,
            random_seed=args.seed,
            command=command,
            image=args.image,
            timeout_seconds=args.trial_timeout,
            allocated_gpus=args.gpus,
            network=args.network,
            memory=args.memory,
            cpus=args.cpus,
        )
    )
    return 0


def _cmd_recovery_fault_list(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.task)
    _print_json(
        {
            "task_id": manifest.task_id,
            "faults": [
                {"fault_id": fault.fault_id, "family": fault.family, "title": fault.title}
                for fault in task_faults(manifest)
            ],
        }
    )
    return 0


def _cmd_recovery_diagnose(args: argparse.Namespace) -> int:
    _print_json(record_recovery_diagnosis(args.run_dir, list(args.labels)))
    return 0


def _cmd_remote_recovery_diagnose(args: argparse.Namespace) -> int:
    _print_json(submit_remote_recovery_diagnosis(args.socket, list(args.labels)))
    return 0


def _recovery_command(args: argparse.Namespace, *, image: str | None) -> int:
    command = list(args.agent_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("recovery runner requires a command after --")
    result = run_recovery_agent(
        task_path=args.task,
        prepared_dir=args.prepared,
        run_dir=args.run_dir,
        agent_id=args.agent_id,
        random_seed=args.seed,
        command=command,
        episodes=args.episodes,
        timeout_seconds=args.trial_timeout,
        allocated_gpus=args.gpus,
        image=image,
        network=getattr(args, "network", "none"),
        memory=getattr(args, "memory", "32g"),
        cpus=getattr(args, "cpus", 8.0),
    )
    _print_json(result)
    return 0


def _cmd_run_recovery_agent(args: argparse.Namespace) -> int:
    return _recovery_command(args, image=None)


def _cmd_run_recovery_agent_docker(args: argparse.Namespace) -> int:
    return _recovery_command(args, image=args.image)


def _cmd_grade_recovery(args: argparse.Namespace) -> int:
    _print_json(grade_recovery_campaign(args.run_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recevobench",
        description="Protocol and scoring tools for recommendation evolution agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a task and sealed evaluation bundle")
    validate.add_argument("--task", required=True, type=Path)
    validate.add_argument("--evaluation", required=True, type=Path)
    validate.set_defaults(func=_cmd_validate)

    validate_run = sub.add_parser("validate-run", help="validate an agent's public run submission")
    validate_run.add_argument("--task", required=True, type=Path)
    validate_run.add_argument("--run", required=True, type=Path)
    validate_run.set_defaults(func=_cmd_validate_run)

    score = sub.add_parser("score", help="produce a single-run scorecard")
    score.add_argument("--task", required=True, type=Path)
    score.add_argument("--evaluation", required=True, type=Path)
    score.add_argument(
        "--artifacts",
        type=Path,
        help="campaign directory containing seed-workspace and candidate snapshots",
    )
    score.add_argument("--output", type=Path)
    score.set_defaults(func=_cmd_score)

    aggregate = sub.add_parser("aggregate", help="aggregate repeated-run scorecards")
    aggregate.add_argument("scorecards", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path)
    aggregate.set_defaults(func=_cmd_aggregate)

    prepare = sub.add_parser("prepare", help="prepare and checksum a raw recommendation dataset")
    prepare.add_argument("--task", required=True, type=Path)
    prepare.add_argument("--raw", required=True, nargs="+", type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--limit", type=int, help="deterministic raw-row limit for smoke tests")
    prepare.set_defaults(func=_cmd_prepare)

    verify_data = sub.add_parser("verify-data", help="verify a prepared dataset card and checksums")
    verify_data.add_argument("--prepared", required=True, type=Path)
    verify_data.set_defaults(func=_cmd_verify_data)

    evaluate = sub.add_parser("evaluate", help="evaluate predictions on one prepared split")
    evaluate.add_argument("--task", required=True, type=Path)
    evaluate.add_argument("--prepared", required=True, type=Path)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(func=_cmd_evaluate)

    execute = sub.add_parser("execute-candidate", help="train, predict and grade one candidate")
    execute.add_argument("--task", required=True, type=Path)
    execute.add_argument("--prepared", required=True, type=Path)
    execute.add_argument("--workspace", required=True, type=Path)
    execute.add_argument("--split", required=True)
    execute.add_argument("--output", required=True, type=Path)
    execute.add_argument("--timeout", type=float, default=3600.0)
    execute.add_argument("--seed", type=int, default=0)
    execute.add_argument("--gpus", type=int, default=1)
    execute.set_defaults(func=_cmd_execute_candidate)

    campaign_init = sub.add_parser("campaign-init", help="create and calibrate a public campaign")
    campaign_init.add_argument("--task", required=True, type=Path)
    campaign_init.add_argument("--prepared", required=True, type=Path)
    campaign_init.add_argument("--run-dir", required=True, type=Path)
    campaign_init.add_argument("--agent-id", required=True)
    campaign_init.add_argument("--seed", required=True, type=int)
    campaign_init.add_argument("--trial-timeout", type=float, default=3600.0)
    campaign_init.add_argument("--gpus", type=int, default=1)
    campaign_init.add_argument("--sandbox-image", help="run candidate code in this Docker image")
    campaign_init.set_defaults(func=_cmd_campaign_init)

    trial = sub.add_parser("trial", help="evaluate and append one public candidate trial")
    trial.add_argument("--run-dir", required=True, type=Path)
    trial.add_argument("--candidate-id", required=True)
    trial.add_argument("--parent-id")
    trial.add_argument("--hypothesis", default="")
    trial.add_argument("--promoted", action="store_true")
    trial.add_argument("--input-tokens", type=int, default=0)
    trial.add_argument("--output-tokens", type=int, default=0)
    trial.add_argument("--human-interventions", type=int, default=0)
    trial.set_defaults(func=_cmd_trial)

    campaign_finalize = sub.add_parser(
        "campaign-finalize", help="freeze an agent-selected candidate"
    )
    campaign_finalize.add_argument("--run-dir", required=True, type=Path)
    campaign_finalize.add_argument("--final-candidate-id")
    campaign_finalize.set_defaults(func=_cmd_campaign_finalize)

    grade_campaign_parser = sub.add_parser(
        "grade-campaign", help="trusted hidden and sealed grading"
    )
    grade_campaign_parser.add_argument("--run-dir", required=True, type=Path)
    grade_campaign_parser.set_defaults(func=_cmd_grade_campaign)

    validate_schema = sub.add_parser(
        "validate-schema", help="validate a JSON artifact against its public schema"
    )
    validate_schema.add_argument("--kind", required=True, choices=sorted(SCHEMA_FILES))
    validate_schema.add_argument("--input", required=True, type=Path)
    validate_schema.set_defaults(func=_cmd_validate_schema)

    task_list = sub.add_parser("task-list", help="list installed benchmark tasks")
    task_list.set_defaults(func=_cmd_task_list)

    smoke = sub.add_parser("smoke", help="run all built-in tasks end to end on generated tiny data")
    smoke.add_argument("--output", required=True, type=Path)
    smoke.set_defaults(func=_cmd_smoke)

    run_agent = sub.add_parser("run-agent", help="run a black-box agent in a calibrated campaign")
    run_agent.add_argument("--task", required=True, type=Path)
    run_agent.add_argument("--prepared", required=True, type=Path)
    run_agent.add_argument("--run-dir", required=True, type=Path)
    run_agent.add_argument("--agent-id", required=True)
    run_agent.add_argument("--seed", required=True, type=int)
    run_agent.add_argument("--trial-timeout", type=float, default=3600.0)
    run_agent.add_argument("--gpus", type=int, default=1)
    run_agent.add_argument("--sandbox-image", help="run candidate code in this Docker image")
    run_agent.add_argument("agent_command", nargs=argparse.REMAINDER)
    run_agent.set_defaults(func=_cmd_run_agent)

    remote_trial = sub.add_parser(
        "remote-trial", help="submit a trial through the host-owned control socket"
    )
    remote_trial.add_argument("--socket", required=True, type=Path)
    remote_trial.add_argument("--candidate-id", required=True)
    remote_trial.add_argument("--parent-id")
    remote_trial.add_argument("--hypothesis", default="")
    remote_trial.add_argument("--promoted", action="store_true")
    remote_trial.add_argument("--input-tokens", type=int, default=0)
    remote_trial.add_argument("--output-tokens", type=int, default=0)
    remote_trial.add_argument("--human-interventions", type=int, default=0)
    remote_trial.set_defaults(func=_cmd_remote_trial)

    remote_finalize = sub.add_parser(
        "remote-finalize", help="select the final candidate through the control socket"
    )
    remote_finalize.add_argument("--socket", required=True, type=Path)
    remote_finalize.add_argument("--candidate-id", required=True)
    remote_finalize.set_defaults(func=_cmd_remote_finalize)

    run_agent_container = sub.add_parser(
        "run-agent-docker",
        help="run an agent and its candidates in isolated Docker containers",
    )
    run_agent_container.add_argument("--task", required=True, type=Path)
    run_agent_container.add_argument("--prepared", required=True, type=Path)
    run_agent_container.add_argument("--run-dir", required=True, type=Path)
    run_agent_container.add_argument("--agent-id", required=True)
    run_agent_container.add_argument("--seed", required=True, type=int)
    run_agent_container.add_argument("--image", required=True)
    run_agent_container.add_argument("--trial-timeout", type=float, default=3600.0)
    run_agent_container.add_argument("--gpus", type=int, default=1)
    run_agent_container.add_argument("--network", choices=("none", "bridge"), default="none")
    run_agent_container.add_argument("--memory", default="32g")
    run_agent_container.add_argument("--cpus", type=float, default=8.0)
    run_agent_container.add_argument("agent_command", nargs=argparse.REMAINDER)
    run_agent_container.set_defaults(func=_cmd_run_agent_docker)

    recovery_faults = sub.add_parser(
        "recovery-fault-list", help="list hidden-fault labels available for a task"
    )
    recovery_faults.add_argument("--task", required=True, type=Path)
    recovery_faults.set_defaults(func=_cmd_recovery_fault_list)

    recovery_diagnose = sub.add_parser(
        "recovery-diagnose", help="submit one diagnosis for the current recovery episode"
    )
    recovery_diagnose.add_argument("--run-dir", required=True, type=Path)
    recovery_diagnose.add_argument("labels", nargs="+", metavar="FAULT_LABEL")
    recovery_diagnose.set_defaults(func=_cmd_recovery_diagnose)

    remote_recovery_diagnose = sub.add_parser(
        "remote-recovery-diagnose",
        help="submit an episode diagnosis through the host control socket",
    )
    remote_recovery_diagnose.add_argument("--socket", required=True, type=Path)
    remote_recovery_diagnose.add_argument("labels", nargs="+", metavar="FAULT_LABEL")
    remote_recovery_diagnose.set_defaults(func=_cmd_remote_recovery_diagnose)

    run_recovery = sub.add_parser(
        "run-recovery-agent",
        help="run a local trusted multi-episode hidden-fault campaign",
    )
    run_recovery.add_argument("--task", required=True, type=Path)
    run_recovery.add_argument("--prepared", required=True, type=Path)
    run_recovery.add_argument("--run-dir", required=True, type=Path)
    run_recovery.add_argument("--agent-id", required=True)
    run_recovery.add_argument("--seed", required=True, type=int)
    run_recovery.add_argument("--episodes", type=int)
    run_recovery.add_argument("--trial-timeout", type=float, default=3600.0)
    run_recovery.add_argument("--gpus", type=int, default=1)
    run_recovery.add_argument("agent_command", nargs=argparse.REMAINDER)
    run_recovery.set_defaults(func=_cmd_run_recovery_agent)

    run_recovery_docker = sub.add_parser(
        "run-recovery-agent-docker",
        help="run an isolated multi-episode hidden-fault campaign",
    )
    run_recovery_docker.add_argument("--task", required=True, type=Path)
    run_recovery_docker.add_argument("--prepared", required=True, type=Path)
    run_recovery_docker.add_argument("--run-dir", required=True, type=Path)
    run_recovery_docker.add_argument("--agent-id", required=True)
    run_recovery_docker.add_argument("--seed", required=True, type=int)
    run_recovery_docker.add_argument("--episodes", type=int)
    run_recovery_docker.add_argument("--image", required=True)
    run_recovery_docker.add_argument("--trial-timeout", type=float, default=3600.0)
    run_recovery_docker.add_argument("--gpus", type=int, default=1)
    run_recovery_docker.add_argument("--network", choices=("none", "bridge"), default="none")
    run_recovery_docker.add_argument("--memory", default="32g")
    run_recovery_docker.add_argument("--cpus", type=float, default=8.0)
    run_recovery_docker.add_argument("agent_command", nargs=argparse.REMAINDER)
    run_recovery_docker.set_defaults(func=_cmd_run_recovery_agent_docker)

    grade_recovery = sub.add_parser(
        "grade-recovery", help="trusted grading for a completed recovery campaign"
    )
    grade_recovery.add_argument("--run-dir", required=True, type=Path)
    grade_recovery.set_defaults(func=_cmd_grade_recovery)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # CLI boundary: render concise machine-friendly failure.
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
