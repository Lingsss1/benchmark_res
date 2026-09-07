"""Built-in adapter registry and task-to-adapter dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..contracts import TaskManifest
from .amazon import AmazonAdapter
from .base import DatasetAdapter, DatasetCard, PreparationError
from .kuairec import KuaiRecAdapter
from .taobao import TaobaoAdapter

ADAPTERS: dict[str, DatasetAdapter] = {
    "amazon": AmazonAdapter(),
    "kuairec": KuaiRecAdapter(),
    "taobao": TaobaoAdapter(),
}


def adapter_name_for_task(manifest: TaskManifest) -> str:
    family = manifest.dataset_family.lower()
    if "amazon" in family:
        return "amazon"
    if "kuairec" in family:
        return "kuairec"
    if "taobao" in family:
        return "taobao"
    configured = manifest.metadata.get("adapter")
    if configured:
        return str(configured)
    raise PreparationError(f"no adapter registered for dataset family {manifest.dataset_family!r}")


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise PreparationError(
            f"unknown adapter {name!r}; available: {', '.join(sorted(ADAPTERS))}"
        ) from exc


def prepare_task(
    manifest: TaskManifest,
    raw_paths: Sequence[Path],
    output_dir: Path,
    *,
    limit: int | None = None,
) -> DatasetCard:
    return get_adapter(adapter_name_for_task(manifest)).prepare(
        task_id=manifest.task_id,
        raw_paths=raw_paths,
        output_dir=output_dir,
        limit=limit,
    )
