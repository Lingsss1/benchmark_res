"""JSON loading and stable artifact writing helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .assets import resolve_task_path
from .contracts import EvaluationBundle, RunTrace, TaskManifest


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_manifest(path: str | Path) -> TaskManifest:
    return TaskManifest.from_dict(read_json(resolve_task_path(path)))


def load_evaluation_bundle(path: str | Path) -> EvaluationBundle:
    return EvaluationBundle.from_dict(read_json(path))


def load_run_trace(path: str | Path) -> RunTrace:
    return RunTrace.from_dict(read_json(path))


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
