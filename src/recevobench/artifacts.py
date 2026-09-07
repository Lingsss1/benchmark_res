"""Stable hashing and editable-path enforcement for candidate workspaces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

DEFAULT_IGNORES = (
    ".git/**",
    "__pycache__/**",
    "*.pyc",
    ".pytest_cache/**",
    "artifacts/**",
    "predictions/**",
)


def path_matches(path: str, patterns: Iterable[str]) -> bool:
    normalized = PurePosixPath(path)
    return any(
        normalized.match(pattern) or (pattern.startswith("**/") and normalized.match(pattern[3:]))
        for pattern in patterns
    )


def file_hashes(root: str | Path, *, ignore: Iterable[str] = DEFAULT_IGNORES) -> dict[str, str]:
    base = Path(root)
    result: dict[str, str] = {}
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if path_matches(relative, ignore):
            continue
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def directory_digest(root: str | Path, *, ignore: Iterable[str] = DEFAULT_IGNORES) -> str:
    hashes = file_hashes(root, ignore=ignore)
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def changed_paths(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    )


def forbidden_changes(
    before: Mapping[str, str],
    after: Mapping[str, str],
    editable_patterns: Iterable[str],
    *,
    always_allowed: Iterable[str] = DEFAULT_IGNORES,
) -> tuple[str, ...]:
    allowed = tuple(editable_patterns) + tuple(always_allowed)
    return tuple(path for path in changed_paths(before, after) if not path_matches(path, allowed))


def total_file_bytes(root: str | Path) -> int:
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())
