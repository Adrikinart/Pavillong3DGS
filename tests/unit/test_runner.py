"""Runner invariants: status transitions, skip/force, failure gating, fingerprint."""

import logging
from pathlib import Path

import pytest

from video_to_3dgs.core.errors import OutputValidationError
from video_to_3dgs.core.manifest import Manifest
from video_to_3dgs.core.paths import RunLayout
from video_to_3dgs.core.runner import StageRunner, topo_sort
from video_to_3dgs.core.stage import Artifact, Stage, StageContext
from video_to_3dgs.core.status import StageStatus, read_status


class _Cfg:
    """Minimal stand-in for PipelineConfig for runner tests."""
    class storage:  # noqa: N801
        scratch_root = "auto"


def _ctx(tmp_path, force=False, dry=False, params=None):
    lay = RunLayout(runs_root=tmp_path, dataset_id="d")
    lay.ensure_base_dirs()
    return StageContext(layout=lay, config=_Cfg(), manifest=Manifest(lay.manifest),
                        logger=logging.getLogger("test"), repo_root=tmp_path,
                        dry_run=dry, force=force, params=params or {})


class WriteFileStage(Stage):
    name = "wf"

    def declared_outputs(self, ctx):
        return [Artifact("out", ctx.layout.run_dir / "out.txt", "file")]

    def stage_params(self, ctx):
        return {"val": ctx.params.get("val", 1)}

    def run(self, ctx):
        (ctx.layout.run_dir / "out.txt").write_text(str(ctx.params.get("val", 1)))
        return {"wrote": True}


class BrokenStage(Stage):
    name = "broken"

    def declared_outputs(self, ctx):
        return [Artifact("missing", ctx.layout.run_dir / "never.txt", "file")]

    def run(self, ctx):
        return {}  # produces nothing -> validate_outputs must fail


def test_completes_and_skips_on_rerun(tmp_path):
    ctx = _ctx(tmp_path)
    r = StageRunner(ctx)
    assert r.execute(WriteFileStage()) == StageStatus.COMPLETED
    rec = read_status(ctx.layout.status_file("wf"), "wf")
    assert rec.state == "COMPLETED" and rec.fingerprint
    # rerun -> SKIPPED (fingerprint match)
    assert r.execute(WriteFileStage()) == StageStatus.SKIPPED


def test_param_change_triggers_rerun(tmp_path):
    ctx = _ctx(tmp_path, params={"val": 1})
    StageRunner(ctx).execute(WriteFileStage())
    ctx2 = _ctx(tmp_path, params={"val": 2})
    # different fingerprint -> not skipped
    assert StageRunner(ctx2).execute(WriteFileStage()) == StageStatus.COMPLETED
    assert (tmp_path / "d" / "out.txt").read_text() == "2"


def test_failed_stage_never_completed(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(OutputValidationError):
        StageRunner(ctx).execute(BrokenStage())
    rec = read_status(ctx.layout.status_file("broken"), "broken")
    assert rec.state == "FAILED"
    assert rec.error


def test_dry_run_no_side_effects(tmp_path):
    ctx = _ctx(tmp_path, dry=True)
    StageRunner(ctx).execute(WriteFileStage())
    assert not (tmp_path / "d" / "out.txt").exists()
    rec = read_status(ctx.layout.status_file("wf"), "wf")
    assert rec.state == "PENDING"


def test_force_reruns_completed(tmp_path):
    ctx = _ctx(tmp_path)
    StageRunner(ctx).execute(WriteFileStage())
    ctx_forced = _ctx(tmp_path, force=True)
    assert StageRunner(ctx_forced).execute(WriteFileStage()) == StageStatus.COMPLETED


def test_topo_sort_orders_dependencies():
    class A(Stage):
        name = "a"

    class B(Stage):
        name = "b"
        depends_on = ("a",)

    class C(Stage):
        name = "c"
        depends_on = ("b",)

    order = [s.name for s in topo_sort([C(), A(), B()])]
    assert order.index("a") < order.index("b") < order.index("c")
