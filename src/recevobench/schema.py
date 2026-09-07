"""JSON Schema discovery and validation for public benchmark artifacts."""

from __future__ import annotations

import sysconfig
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_FILES = {
    "task": "task-manifest.schema.json",
    "run": "public-run.schema.json",
    "dataset": "dataset-card.schema.json",
    "evaluation": "evaluation-bundle.schema.json",
    "scorecard": "scorecard.schema.json",
}


def schema_directory() -> Path:
    repository = Path(__file__).resolve().parents[2] / "schemas"
    if repository.is_dir():
        return repository
    installed = Path(sysconfig.get_path("data")) / "share" / "recevobench" / "schemas"
    if installed.is_dir():
        return installed
    raise FileNotFoundError("RecEvoBench JSON schemas are not installed")


def validate_artifact(kind: str, value: Any) -> None:
    try:
        filename = SCHEMA_FILES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown schema kind {kind!r}") from exc
    import json

    schema = json.loads((schema_directory() / filename).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(value)
