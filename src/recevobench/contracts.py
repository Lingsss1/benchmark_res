"""Typed contracts shared by benchmark tasks, agents, and sealed evaluators.

The benchmark treats an evolution agent as a black box.  Agents may use any
search policy, memory system, or code-generation approach, but every run must
emit the same public trace.  Hidden selection and test measurements are added
by the benchmark evaluator after the agent can no longer use them as feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Direction = Literal["maximize", "minimize"]
TrialStatus = Literal[
    "valid",
    "crash",
    "oom",
    "timeout",
    "protocol_violation",
    "invalid",
]


class ContractError(ValueError):
    """Raised when a manifest or run artifact violates the benchmark schema."""


def _required(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ContractError(f"missing {where}.{key}")
    return data[key]


def _finite_float(value: Any, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{where} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ContractError(f"{where} must be finite")
    return result


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: Direction
    lower_bound: float | None = None
    upper_bound: float | None = None
    absolute_epsilon: float = 0.0
    relative_epsilon: float = 0.005

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricSpec:
        name = str(_required(data, "name", "primary_metric")).strip()
        direction = str(_required(data, "direction", "primary_metric"))
        if not name:
            raise ContractError("primary_metric.name cannot be empty")
        if direction not in {"maximize", "minimize"}:
            raise ContractError("primary_metric.direction must be maximize or minimize")
        lower = data.get("lower_bound")
        upper = data.get("upper_bound")
        lower_f = None if lower is None else _finite_float(lower, "primary_metric.lower_bound")
        upper_f = None if upper is None else _finite_float(upper, "primary_metric.upper_bound")
        if lower_f is not None and upper_f is not None and upper_f <= lower_f:
            raise ContractError("primary metric upper_bound must exceed lower_bound")
        return cls(
            name=name,
            direction=direction,  # type: ignore[arg-type]
            lower_bound=lower_f,
            upper_bound=upper_f,
            absolute_epsilon=max(
                0.0,
                _finite_float(data.get("absolute_epsilon", 0.0), "primary_metric.absolute_epsilon"),
            ),
            relative_epsilon=max(
                0.0,
                _finite_float(
                    data.get("relative_epsilon", 0.005),
                    "primary_metric.relative_epsilon",
                ),
            ),
        )

    def orient(self, value: float) -> float:
        """Map the native metric to a higher-is-better value."""
        return value if self.direction == "maximize" else -value

    @property
    def scale(self) -> float | None:
        if self.lower_bound is None or self.upper_bound is None:
            return None
        return self.upper_bound - self.lower_bound

    def improvement_threshold(self, baseline: float) -> float:
        return max(self.absolute_epsilon, abs(baseline) * self.relative_epsilon)


@dataclass(frozen=True)
class GuardrailSpec:
    metric: str
    direction: Direction
    max_absolute_degradation: float = 0.0
    max_relative_degradation: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardrailSpec:
        metric = str(_required(data, "metric", "guardrail")).strip()
        direction = str(_required(data, "direction", "guardrail"))
        if not metric:
            raise ContractError("guardrail.metric cannot be empty")
        if direction not in {"maximize", "minimize"}:
            raise ContractError("guardrail.direction must be maximize or minimize")
        return cls(
            metric=metric,
            direction=direction,  # type: ignore[arg-type]
            max_absolute_degradation=max(
                0.0,
                _finite_float(
                    data.get("max_absolute_degradation", 0.0),
                    f"guardrail[{metric}].max_absolute_degradation",
                ),
            ),
            max_relative_degradation=max(
                0.0,
                _finite_float(
                    data.get("max_relative_degradation", 0.0),
                    f"guardrail[{metric}].max_relative_degradation",
                ),
            ),
        )


@dataclass(frozen=True)
class BudgetSpec:
    max_compute_units: float
    max_trials: int
    max_wall_seconds: float | None = None
    max_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetSpec:
        compute = _finite_float(
            _required(data, "max_compute_units", "budget"),
            "budget.max_compute_units",
        )
        trials = int(_required(data, "max_trials", "budget"))
        if compute <= 0 or trials <= 0:
            raise ContractError("budget limits must be positive")
        wall = data.get("max_wall_seconds")
        tokens = data.get("max_tokens")
        wall_f = None if wall is None else _finite_float(wall, "budget.max_wall_seconds")
        tokens_i = None if tokens is None else int(tokens)
        if wall_f is not None and wall_f <= 0:
            raise ContractError("budget.max_wall_seconds must be positive")
        if tokens_i is not None and tokens_i <= 0:
            raise ContractError("budget.max_tokens must be positive")
        return cls(compute, trials, wall_f, tokens_i)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    visibility: Literal["public", "hidden", "sealed"]
    purpose: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitSpec:
        name = str(_required(data, "name", "split")).strip()
        visibility = str(_required(data, "visibility", "split"))
        if visibility not in {"public", "hidden", "sealed"}:
            raise ContractError(f"invalid visibility for split {name!r}")
        return cls(name, visibility, str(data.get("purpose", "")))  # type: ignore[arg-type]


@dataclass(frozen=True)
class TaskManifest:
    schema_version: str
    task_id: str
    title: str
    dataset_family: str
    task_type: str
    primary_metric: MetricSpec
    secondary_metrics: tuple[str, ...]
    guardrails: tuple[GuardrailSpec, ...]
    budget: BudgetSpec
    splits: tuple[SplitSpec, ...]
    evaluation_protocol: dict[str, Any]
    editable_paths: tuple[str, ...]
    recovery_protocol: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    artifact_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskManifest:
        splits = tuple(SplitSpec.from_dict(x) for x in _required(data, "splits", "task"))
        visibilities = {split.visibility for split in splits}
        if "public" not in visibilities or "sealed" not in visibilities:
            raise ContractError("task must define at least one public and one sealed split")
        manifest = cls(
            schema_version=str(data.get("schema_version", "1.0")),
            task_id=str(_required(data, "task_id", "task")).strip(),
            title=str(_required(data, "title", "task")).strip(),
            dataset_family=str(_required(data, "dataset_family", "task")).strip(),
            task_type=str(_required(data, "task_type", "task")).strip(),
            primary_metric=MetricSpec.from_dict(_required(data, "primary_metric", "task")),
            secondary_metrics=tuple(str(x) for x in data.get("secondary_metrics", [])),
            guardrails=tuple(GuardrailSpec.from_dict(x) for x in data.get("guardrails", [])),
            budget=BudgetSpec.from_dict(_required(data, "budget", "task")),
            splits=splits,
            evaluation_protocol=dict(_required(data, "evaluation_protocol", "task")),
            editable_paths=tuple(str(x) for x in data.get("editable_paths", [])),
            recovery_protocol=dict(data.get("recovery_protocol", {})),
            runtime=dict(data.get("runtime", {})),
            artifact_policy=dict(data.get("artifact_policy", {})),
            metadata=dict(data.get("metadata", {})),
        )
        if not manifest.task_id or not manifest.title:
            raise ContractError("task_id and title cannot be empty")
        return manifest


@dataclass(frozen=True)
class TrialRecord:
    step: int
    candidate_id: str
    parent_id: str | None
    status: TrialStatus
    public_metric: float | None
    hidden_select_metric: float | None
    gpu_seconds: float
    wall_seconds: float
    compute_seconds: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    promoted: bool = False
    hypothesis: str = ""
    human_interventions: int = 0
    candidate_sha256: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialRecord:
        status = str(_required(data, "status", "trial"))
        allowed = {"valid", "crash", "oom", "timeout", "protocol_violation", "invalid"}
        if status not in allowed:
            raise ContractError(f"unknown trial status {status!r}")
        public = data.get("public_metric")
        hidden = data.get("hidden_select_metric")
        trial = cls(
            step=int(_required(data, "step", "trial")),
            candidate_id=str(_required(data, "candidate_id", "trial")).strip(),
            parent_id=None if data.get("parent_id") is None else str(data["parent_id"]),
            status=status,  # type: ignore[arg-type]
            public_metric=None if public is None else _finite_float(public, "trial.public_metric"),
            hidden_select_metric=(
                None if hidden is None else _finite_float(hidden, "trial.hidden_select_metric")
            ),
            gpu_seconds=max(0.0, _finite_float(data.get("gpu_seconds", 0.0), "trial.gpu_seconds")),
            wall_seconds=max(
                0.0,
                _finite_float(data.get("wall_seconds", 0.0), "trial.wall_seconds"),
            ),
            compute_seconds=(
                None
                if data.get("compute_seconds") is None
                else max(
                    0.0,
                    _finite_float(data["compute_seconds"], "trial.compute_seconds"),
                )
            ),
            input_tokens=max(0, int(data.get("input_tokens", 0))),
            output_tokens=max(0, int(data.get("output_tokens", 0))),
            promoted=bool(data.get("promoted", False)),
            hypothesis=str(data.get("hypothesis", "")),
            human_interventions=max(0, int(data.get("human_interventions", 0))),
            candidate_sha256=(
                None if data.get("candidate_sha256") is None else str(data["candidate_sha256"])
            ),
        )
        if trial.step <= 0 or not trial.candidate_id:
            raise ContractError("trial.step must be positive and candidate_id cannot be empty")
        if trial.status == "valid" and trial.public_metric is None:
            raise ContractError("valid trial must contain public_metric")
        return trial


@dataclass(frozen=True)
class RunTrace:
    run_id: str
    task_id: str
    agent_id: str
    random_seed: int
    final_candidate_id: str
    baseline_compute_seconds: float
    elapsed_wall_seconds: float
    trials: tuple[TrialRecord, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunTrace:
        trials = tuple(TrialRecord.from_dict(x) for x in _required(data, "trials", "run"))
        steps = [trial.step for trial in trials]
        if steps != sorted(steps) or len(set(steps)) != len(steps):
            raise ContractError("trial steps must be unique and sorted")
        candidate_ids = [trial.candidate_id for trial in trials]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ContractError("candidate_id must be unique within a run")
        earlier = {"seed"}
        for trial in trials:
            if trial.parent_id is not None and trial.parent_id not in earlier:
                raise ContractError(f"trial {trial.candidate_id!r} references a non-earlier parent")
            earlier.add(trial.candidate_id)
        baseline_compute = _finite_float(
            _required(data, "baseline_compute_seconds", "run"),
            "run.baseline_compute_seconds",
        )
        if baseline_compute <= 0:
            raise ContractError("run.baseline_compute_seconds must be positive")
        elapsed_wall = _finite_float(
            _required(data, "elapsed_wall_seconds", "run"),
            "run.elapsed_wall_seconds",
        )
        if elapsed_wall < 0:
            raise ContractError("run.elapsed_wall_seconds cannot be negative")
        final_candidate = str(_required(data, "final_candidate_id", "run"))
        if final_candidate != "seed" and final_candidate not in candidate_ids:
            raise ContractError("final_candidate_id is not present in the trajectory")
        return cls(
            run_id=str(_required(data, "run_id", "run")),
            task_id=str(_required(data, "task_id", "run")),
            agent_id=str(_required(data, "agent_id", "run")),
            random_seed=int(_required(data, "random_seed", "run")),
            final_candidate_id=final_candidate,
            baseline_compute_seconds=baseline_compute,
            elapsed_wall_seconds=elapsed_wall,
            trials=trials,
        )


@dataclass(frozen=True)
class FaultEvaluation:
    fault_id: str
    target_label: str
    predicted_labels: tuple[str, ...]
    repaired: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaultEvaluation:
        return cls(
            fault_id=str(_required(data, "fault_id", "fault")),
            target_label=str(_required(data, "target_label", "fault")),
            predicted_labels=tuple(str(x) for x in data.get("predicted_labels", [])),
            repaired=bool(data.get("repaired", False)),
        )


@dataclass(frozen=True)
class EvaluationBundle:
    """A run trace plus measurements produced by the sealed evaluator."""

    run: RunTrace
    baseline_hidden_select_metrics: dict[str, float]
    baseline_test_metrics: dict[str, float]
    final_test_metrics: dict[str, float]
    baseline_guardrail_metrics: dict[str, float]
    final_guardrail_metrics: dict[str, float]
    faults: tuple[FaultEvaluation, ...] = ()
    baseline_public_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationBundle:
        def metric_map(key: str) -> dict[str, float]:
            raw = dict(_required(data, key, "evaluation"))
            return {
                name: _finite_float(value, f"evaluation.{key}.{name}")
                for name, value in raw.items()
            }

        return cls(
            run=RunTrace.from_dict(_required(data, "run", "evaluation")),
            baseline_hidden_select_metrics=metric_map("baseline_hidden_select_metrics"),
            baseline_test_metrics=metric_map("baseline_test_metrics"),
            final_test_metrics=metric_map("final_test_metrics"),
            baseline_guardrail_metrics={
                name: _finite_float(value, f"evaluation.baseline_guardrail_metrics.{name}")
                for name, value in dict(data.get("baseline_guardrail_metrics", {})).items()
            },
            final_guardrail_metrics={
                name: _finite_float(value, f"evaluation.final_guardrail_metrics.{name}")
                for name, value in dict(data.get("final_guardrail_metrics", {})).items()
            },
            faults=tuple(FaultEvaluation.from_dict(x) for x in data.get("faults", [])),
            baseline_public_metrics={
                name: _finite_float(value, f"evaluation.baseline_public_metrics.{name}")
                for name, value in dict(data.get("baseline_public_metrics", {})).items()
            },
        )
