"""Locate built-in tasks, schemas and seed workspaces in source or wheel installs."""

from __future__ import annotations

import sysconfig
from pathlib import Path


def asset_roots() -> tuple[Path, ...]:
    source_root = Path(__file__).resolve().parents[2]
    installed_root = Path(sysconfig.get_path("data")) / "share" / "recevobench"
    return (source_root, installed_root)


def resolve_asset(relative: str | Path) -> Path:
    value = Path(relative)
    if value.is_absolute() and value.exists():
        return value.resolve()
    for root in asset_roots():
        candidate = root / value
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"RecEvoBench asset not found: {relative}")


def resolve_task_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    name = path.name
    if not name.endswith(".json"):
        name += ".json"
    return resolve_asset(Path("tasks") / name)


def list_builtin_tasks() -> tuple[str, ...]:
    for root in asset_roots():
        directory = root / "tasks"
        if directory.is_dir():
            return tuple(sorted(path.stem for path in directory.glob("*.json")))
    return ()
