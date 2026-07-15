from pathlib import Path

import pytest

from video_to_3dgs.core.atomicio import (
    atomic_write_json,
    canonical_json,
    iter_jsonl,
    append_jsonl,
    read_json,
    sha256_str,
)
from video_to_3dgs.core.manifest import Manifest
from video_to_3dgs.core.paths import RunLayout, resolve_scratch_root, slugify
from video_to_3dgs.core.status import (
    StageStatus,
    StatusRecord,
    process_alive,
    read_status,
    write_status,
)


def test_slugify():
    assert slugify("Casque Saint-Georges!") == "casque_saint_georges"
    assert slugify("  ") == "dataset"


def test_run_layout_paths(tmp_path):
    lay = RunLayout(runs_root=tmp_path, dataset_id="obj_123")
    assert lay.run_dir == tmp_path / "obj_123"
    assert lay.status_file("train").name == "train.json"
    assert lay.checkpoints_dir("t1") == tmp_path / "obj_123" / "trainings" / "t1" / "checkpoints"
    assert lay.split_file("val").name == "val.txt"


def test_scratch_root_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert resolve_scratch_root("auto") == tmp_path
    explicit = tmp_path / "x"
    explicit.mkdir()
    assert resolve_scratch_root(str(explicit)) == explicit


def test_atomic_json_roundtrip(tmp_path):
    p = tmp_path / "a" / "b.json"
    atomic_write_json(p, {"k": 1})
    assert read_json(p) == {"k": 1}


def test_canonical_json_sorted():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert sha256_str("x").startswith("sha256:")


def test_jsonl_tolerates_torn_tail(tmp_path):
    p = tmp_path / "m.jsonl"
    append_jsonl(p, {"i": 0})
    append_jsonl(p, {"i": 1})
    with open(p, "a") as f:
        f.write('{"i": 2, "broken"')  # torn line
    recs = list(iter_jsonl(p))
    assert [r["i"] for r in recs] == [0, 1]


def test_status_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    rec = StatusRecord(stage="train", state=StageStatus.COMPLETED.value, metrics={"psnr": 30})
    write_status(p, rec)
    got = read_status(p, "train")
    assert got.state == "COMPLETED"
    assert got.metrics["psnr"] == 30


def test_status_missing_returns_pending(tmp_path):
    got = read_status(tmp_path / "nope.json", "x")
    assert got.state == StageStatus.PENDING.value


def test_process_alive_dead_pid():
    # pid 2**30 almost certainly does not exist
    assert process_alive(2**30, None) is False


def test_manifest_header_and_append(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.init_header(dataset_id="d1", videos=[{"path": "v.mov"}], software={"torch": "x"})
    m.append_stage({"stage": "extract_frames", "state": "COMPLETED"})
    m.append_stage({"stage": "run_colmap", "state": "COMPLETED"})
    data = m.load()
    assert data["dataset_id"] == "d1"
    assert len(data["stages"]) == 2
    assert data["stages"][0]["stage"] == "extract_frames"
