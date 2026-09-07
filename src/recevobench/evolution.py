"""Objective process metrics for iterative model-evolution trajectories."""

from __future__ import annotations

import ast
import hashlib
import json
import tokenize
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .artifacts import path_matches
from .contracts import EvaluationBundle, MetricSpec, TaskManifest, TrialRecord
from .evaluation.metrics import spearman_correlation


@dataclass(frozen=True)
class EvolutionProcessMetrics:
    validation_test_gap: float | None
    public_hidden_spearman: float | None
    productive_trial_ratio: float
    feedback_conversion_rate: float | None
    misleading_feedback_rate: float | None
    successful_revision_ratio: float | None
    iterative_revision_ratio: float
    post_failure_recovery_rate: float | None
    regression_recovery_rate: float | None
    wasted_compute_fraction: float
    best_improvement_compute_fraction: float | None
    hidden_improvement_per_compute_unit: float | None
    max_lineage_depth: int
    search_branching_ratio: float
    unique_candidate_ratio: float | None
    successful_revision_count: int
    successful_iterative_revision_count: int


@dataclass(frozen=True)
class CodeExplorationMetrics:
    code_exploration_reach: float | None
    code_exploration_spread: float | None
    structural_unique_candidate_ratio: float | None


def _significant_improvement(metric: MetricSpec, parent: float, child: float) -> bool:
    return metric.orient(child) - metric.orient(parent) > metric.improvement_threshold(parent)


def _trial_compute(trial: TrialRecord) -> float:
    return trial.compute_seconds if trial.compute_seconds is not None else trial.gpu_seconds


def _parent_value(
    trial: TrialRecord,
    values: dict[str, float],
    baseline: float | None,
) -> float | None:
    if trial.parent_id in (None, "seed"):
        return baseline
    return values.get(trial.parent_id)


def _lineage_depths(trials: tuple[TrialRecord, ...]) -> dict[str, int]:
    depths = {"seed": 0}
    for trial in trials:
        parent_depth = depths.get(trial.parent_id or "seed", 0)
        depths[trial.candidate_id] = parent_depth + 1
    return depths


def _is_descendant(
    candidate_id: str,
    ancestor_id: str,
    parents: dict[str, str | None],
) -> bool:
    current = candidate_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        parent = parents.get(current)
        if parent == ancestor_id:
            return True
        if parent in (None, "seed"):
            return False
        current = parent
    return False


def _selected_hidden_value(bundle: EvaluationBundle, name: str) -> float | None:
    if bundle.run.final_candidate_id == "seed":
        return bundle.baseline_hidden_select_metrics.get(name)
    for trial in bundle.run.trials:
        if trial.candidate_id == bundle.run.final_candidate_id:
            return trial.hidden_select_metric
    return None


def analyze_evolution_process(
    manifest: TaskManifest, bundle: EvaluationBundle
) -> EvolutionProcessMetrics:
    """Measure adaptation from evaluator-owned metrics rather than self-reported prose."""
    metric = manifest.primary_metric
    name = metric.name
    trials = bundle.run.trials
    valid = [
        trial
        for trial in trials
        if trial.status == "valid" and trial.hidden_select_metric is not None
    ]
    public_values = {
        trial.candidate_id: trial.public_metric
        for trial in valid
        if trial.public_metric is not None
    }
    hidden_values = {
        trial.candidate_id: trial.hidden_select_metric
        for trial in valid
        if trial.hidden_select_metric is not None
    }
    baseline_public = bundle.baseline_public_metrics.get(name)
    baseline_hidden = bundle.baseline_hidden_select_metrics.get(name)
    parents = {trial.candidate_id: trial.parent_id for trial in trials}

    paired = [
        (float(trial.public_metric), float(trial.hidden_select_metric))
        for trial in valid
        if trial.public_metric is not None and trial.hidden_select_metric is not None
    ]
    public_hidden = (
        spearman_correlation(
            [metric.orient(public) for public, _ in paired],
            [metric.orient(hidden) for _, hidden in paired],
        )
        if len(paired) >= 2
        else None
    )

    productive = 0
    best_hidden = baseline_hidden
    productive_ids: set[str] = set()
    for trial in trials:
        hidden = trial.hidden_select_metric
        if trial.status != "valid" or hidden is None:
            continue
        if best_hidden is None or _significant_improvement(metric, best_hidden, hidden):
            productive += 1
            productive_ids.add(trial.candidate_id)
            best_hidden = hidden

    public_improvements = 0
    converted = 0
    successful_revisions = 0
    successful_iterative_revisions = 0
    evaluable_revisions = 0
    hidden_regressions: list[tuple[TrialRecord, float]] = []
    for trial in valid:
        parent_public = _parent_value(trial, public_values, baseline_public)
        parent_hidden = _parent_value(trial, hidden_values, baseline_hidden)
        if parent_hidden is not None:
            evaluable_revisions += 1
            child_hidden = float(trial.hidden_select_metric)
            if _significant_improvement(metric, parent_hidden, child_hidden):
                successful_revisions += 1
                on_final_lineage = (
                    trial.candidate_id == bundle.run.final_candidate_id
                    or _is_descendant(
                        bundle.run.final_candidate_id,
                        trial.candidate_id,
                        parents,
                    )
                )
                if trial.parent_id not in (None, "seed") and on_final_lineage:
                    successful_iterative_revisions += 1
            elif _significant_improvement(metric, child_hidden, parent_hidden):
                hidden_regressions.append((trial, parent_hidden))
        if parent_public is None or parent_hidden is None or trial.public_metric is None:
            continue
        public_better = _significant_improvement(metric, parent_public, float(trial.public_metric))
        if public_better:
            public_improvements += 1
            if _significant_improvement(metric, parent_hidden, float(trial.hidden_select_metric)):
                converted += 1

    recovered_regressions = 0
    for regression, prior_value in hidden_regressions:
        if any(
            later.status == "valid"
            and later.hidden_select_metric is not None
            and _is_descendant(later.candidate_id, regression.candidate_id, parents)
            and _significant_improvement(metric, prior_value, float(later.hidden_select_metric))
            for later in trials
            if later.step > regression.step
        ):
            recovered_regressions += 1

    failures = [index for index, trial in enumerate(trials) if trial.status != "valid"]
    recovered_failures = sum(
        index + 1 < len(trials) and trials[index + 1].status == "valid" for index in failures
    )

    total_compute = sum(_trial_compute(trial) for trial in trials)
    wasted_compute = sum(
        _trial_compute(trial)
        for trial in trials
        if trial.status != "valid" or trial.candidate_id not in productive_ids
    )
    used_compute = 0.0
    best_at = None
    best_value = baseline_hidden
    for trial in trials:
        used_compute += _trial_compute(trial)
        hidden = trial.hidden_select_metric
        if (
            trial.status == "valid"
            and hidden is not None
            and (best_value is None or _significant_improvement(metric, best_value, hidden))
        ):
            best_value = hidden
            best_at = used_compute
    maximum_compute = manifest.budget.max_compute_units * bundle.run.baseline_compute_seconds
    best_fraction = None if best_at is None else min(1.0, best_at / maximum_compute)

    hidden_yield = None
    if baseline_hidden is not None and best_value is not None and total_compute > 0:
        scale = metric.scale
        gain = metric.orient(best_value) - metric.orient(baseline_hidden)
        normalized_gain = gain / scale if scale else gain
        compute_units = total_compute / bundle.run.baseline_compute_seconds
        hidden_yield = max(0.0, normalized_gain) / compute_units

    validation_test_gap = None
    selected_hidden = _selected_hidden_value(bundle, name)
    baseline_test = bundle.baseline_test_metrics.get(name)
    final_test = bundle.final_test_metrics.get(name)
    if (
        baseline_hidden is not None
        and selected_hidden is not None
        and baseline_test is not None
        and final_test is not None
    ):
        scale = metric.scale or 1.0
        validation_gain = (metric.orient(selected_hidden) - metric.orient(baseline_hidden)) / scale
        test_gain = (metric.orient(final_test) - metric.orient(baseline_test)) / scale
        validation_test_gap = abs(validation_gain - test_gain)

    depths = _lineage_depths(trials)
    child_counts = Counter(trial.parent_id or "seed" for trial in trials)
    branching = (
        sum(count > 1 for count in child_counts.values()) / len(child_counts)
        if child_counts
        else 0.0
    )
    hashes = [trial.candidate_sha256 for trial in valid if trial.candidate_sha256]
    iterative = sum(trial.parent_id not in (None, "seed") for trial in valid)
    return EvolutionProcessMetrics(
        validation_test_gap=validation_test_gap,
        public_hidden_spearman=public_hidden,
        productive_trial_ratio=productive / len(valid) if valid else 0.0,
        feedback_conversion_rate=(converted / public_improvements if public_improvements else None),
        misleading_feedback_rate=(
            (public_improvements - converted) / public_improvements if public_improvements else None
        ),
        successful_revision_ratio=(
            successful_revisions / evaluable_revisions if evaluable_revisions else None
        ),
        iterative_revision_ratio=iterative / len(valid) if valid else 0.0,
        post_failure_recovery_rate=(recovered_failures / len(failures) if failures else None),
        regression_recovery_rate=(
            recovered_regressions / len(hidden_regressions) if hidden_regressions else None
        ),
        wasted_compute_fraction=wasted_compute / total_compute if total_compute else 0.0,
        best_improvement_compute_fraction=best_fraction,
        hidden_improvement_per_compute_unit=hidden_yield,
        max_lineage_depth=max(depths.values(), default=0),
        search_branching_ratio=branching,
        unique_candidate_ratio=len(set(hashes)) / len(hashes) if hashes else None,
        successful_revision_count=successful_revisions,
        successful_iterative_revision_count=successful_iterative_revisions,
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}".strip(".")
    return type(node).__name__


def _python_features(path: Path, relative: str) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        with tokenize.open(path) as handle:
            content = handle.read()
        return {
            f"{relative}:raw:{hashlib.sha256(line.strip().encode()).hexdigest()}"
            for line in content.splitlines()
            if line.strip()
        }
    features: set[str] = set()
    for node in ast.walk(tree):
        features.add(f"{relative}:node:{type(node).__name__}")
        if isinstance(node, ast.Call):
            features.add(f"{relative}:call:{_call_name(node.func)}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            features.add(f"{relative}:definition:{type(node).__name__}:{node.name}")
        if isinstance(node, ast.stmt):
            dumped = ast.dump(node, annotate_fields=True, include_attributes=False)
            digest = hashlib.sha256(dumped.encode()).hexdigest()
            features.add(f"{relative}:statement:{digest}")
    return features


def code_structure_features(root: Path, editable_patterns: Iterable[str]) -> frozenset[str]:
    features: set[str] = set()
    patterns = tuple(editable_patterns)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if not path_matches(relative, patterns):
            continue
        if path.suffix == ".py":
            features.update(_python_features(path, relative))
        elif path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                canonical = path.read_bytes().hex()
            features.add(f"{relative}:json:{hashlib.sha256(canonical.encode()).hexdigest()}")
        else:
            features.add(f"{relative}:file:{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return frozenset(features)


def _jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def analyze_code_exploration(
    manifest: TaskManifest,
    bundle: EvaluationBundle,
    artifact_root: str | Path | None,
) -> CodeExplorationMetrics:
    if artifact_root is None:
        return CodeExplorationMetrics(None, None, None)
    root = Path(artifact_root)
    seed = root / "seed-workspace"
    if not seed.is_dir():
        return CodeExplorationMetrics(None, None, None)
    candidate_features: list[frozenset[str]] = []
    for trial in bundle.run.trials:
        candidate = root / "candidates" / trial.candidate_id
        if trial.status == "valid" and candidate.is_dir():
            candidate_features.append(code_structure_features(candidate, manifest.editable_paths))
    if not candidate_features:
        return CodeExplorationMetrics(None, None, None)
    seed_features = code_structure_features(seed, manifest.editable_paths)
    reach = max(_jaccard_distance(seed_features, value) for value in candidate_features)
    pairs = [
        _jaccard_distance(left, right)
        for index, left in enumerate(candidate_features)
        for right in candidate_features[index + 1 :]
    ]
    fingerprints = {
        hashlib.sha256("\n".join(sorted(value)).encode()).hexdigest()
        for value in candidate_features
    }
    return CodeExplorationMetrics(
        code_exploration_reach=reach,
        code_exploration_spread=sum(pairs) / len(pairs) if pairs else 0.0,
        structural_unique_candidate_ratio=len(fingerprints) / len(candidate_features),
    )
