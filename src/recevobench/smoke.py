"""Self-contained end-to-end smoke suite for clean installations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .campaign import evaluate_trial, finalize_campaign, grade_campaign, initialize_campaign
from .data import prepare_task
from .io import load_manifest

AMAZON_RAW = """\
{"user_id":"u1","parent_asin":"i1","timestamp":1,"rating":5}
{"user_id":"u1","parent_asin":"i2","timestamp":2,"rating":4}
{"user_id":"u1","parent_asin":"i3","timestamp":3,"rating":5}
{"user_id":"u1","parent_asin":"i4","timestamp":4,"rating":5}
{"user_id":"u1","parent_asin":"i5","timestamp":5,"rating":4}
{"user_id":"u1","parent_asin":"i6","timestamp":6,"rating":5}
{"user_id":"u2","parent_asin":"i2","timestamp":1,"rating":5}
{"user_id":"u2","parent_asin":"i3","timestamp":2,"rating":5}
{"user_id":"u2","parent_asin":"i4","timestamp":3,"rating":4}
{"user_id":"u2","parent_asin":"i5","timestamp":4,"rating":5}
{"user_id":"u2","parent_asin":"i6","timestamp":5,"rating":4}
{"user_id":"u2","parent_asin":"i1","timestamp":6,"rating":5}
"""

KUAIREC_BIG = """\
user_id,video_id,play_duration,video_duration,timestamp,watch_ratio
u1,v1,10,10,1,1.0
u1,v2,20,10,2,2.0
u1,v3,5,10,3,0.5
u1,v4,30,10,4,3.0
u2,v2,10,10,1,1.0
u2,v3,15,10,2,1.5
u2,v4,25,10,3,2.5
u2,v1,5,10,4,0.5
"""

KUAIREC_SMALL = """\
user_id,video_id,play_duration,video_duration,timestamp,watch_ratio
u1,v1,10,10,10,1.0
u1,v2,20,10,11,2.0
u1,v3,5,10,12,0.5
u1,v4,30,10,13,3.0
u2,v1,5,10,10,0.5
u2,v2,10,10,11,1.0
u2,v3,15,10,12,1.5
u2,v4,25,10,13,2.5
"""

TAOBAO_RAW = """\
user_id,item_id,category_id,behavior_type,timestamp
u1,i1,c1,pv,1
u1,i2,c1,buy,10
u2,i2,c1,pv,20
u2,i3,c2,buy,30
u1,i3,c2,cart,50
u2,i1,c1,fav,69
u1,i4,c2,buy,72
u2,i4,c2,pv,75
u1,i2,c1,buy,82
u2,i3,c2,buy,85
u1,i3,c2,buy,92
u2,i1,c1,pv,93
u2,i4,c2,buy,98
"""


def _write_raw(root: Path, task_id: str) -> list[Path]:
    raw = root / "raw"
    raw.mkdir(parents=True)
    if task_id.startswith("amazon"):
        path = raw / "reviews.jsonl"
        path.write_text(AMAZON_RAW, encoding="utf-8")
        return [path]
    if task_id.startswith("kuairec"):
        (raw / "big_matrix.csv").write_text(KUAIREC_BIG, encoding="utf-8")
        (raw / "small_matrix.csv").write_text(KUAIREC_SMALL, encoding="utf-8")
        return [raw]
    path = raw / "UserBehavior.csv"
    path.write_text(TAOBAO_RAW, encoding="utf-8")
    return [path]


def run_smoke_suite(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    if root.exists():
        raise ValueError(f"smoke output already exists: {root}")
    root.mkdir(parents=True)
    task_ids = (
        "amazon-seqrec-v0",
        "kuairec-watchtime-v0",
        "taobao-purchase-v0",
    )
    results: dict[str, Any] = {}
    for index, task_id in enumerate(task_ids, start=1):
        task_root = root / task_id
        manifest = load_manifest(task_id)
        raw_paths = _write_raw(task_root, task_id)
        prepared = task_root / "prepared"
        prepare_task(manifest, raw_paths, prepared)
        run_dir = task_root / "run"
        initialize_campaign(
            task_path=task_id,
            prepared_dir=prepared,
            run_dir=run_dir,
            agent_id="conformance-agent",
            random_seed=index,
            timeout_seconds=30,
            allocated_gpus=0,
        )
        evaluate_trial(
            run_dir=run_dir,
            candidate_id="conformance-001",
            parent_id="seed",
            hypothesis="No-op protocol conformance check.",
            promoted=True,
        )
        finalize_campaign(run_dir=run_dir, final_candidate_id="conformance-001")
        _, scorecard = grade_campaign(run_dir)
        results[task_id] = {
            "ok": True,
            "primary_metric": scorecard.primary_metric,
            "baseline": scorecard.baseline_test_metric,
            "final": scorecard.final_test_metric,
            "scorecard": str(run_dir / "scorecard.json"),
        }
    return {"ok": True, "tasks": results}
