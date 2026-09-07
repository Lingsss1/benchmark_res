from pathlib import Path

from recevobench.io import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_all_task_manifests_are_valid_and_have_three_visibility_layers() -> None:
    paths = sorted((ROOT / "tasks").glob("*.json"))
    assert {path.stem for path in paths} == {
        "amazon-seqrec-v0",
        "kuairec-watchtime-v0",
        "taobao-purchase-v0",
    }
    for path in paths:
        manifest = load_manifest(path)
        assert {split.visibility for split in manifest.splits} == {
            "public",
            "hidden",
            "sealed",
        }
        assert manifest.budget.max_compute_units > 0
        assert manifest.editable_paths
