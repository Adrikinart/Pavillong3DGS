"""colmap binary resolution (regression for the Slurm PATH pitfall)."""

import pytest

from video_to_3dgs.core.errors import InputValidationError
from video_to_3dgs.stages.run_colmap import resolve_colmap_bin


def test_resolve_explicit_existing_path():
    # an existing absolute path is honored as-is
    assert resolve_colmap_bin("/bin/sh") == "/bin/sh"


def test_resolve_raises_when_absent(monkeypatch, tmp_path):
    # ensure 'colmap' is found neither on PATH nor in <sys.prefix>/bin. The env
    # this suite runs in (v2gs) DOES ship colmap in sys.prefix/bin, so the prefix
    # must be redirected too or the resolver legitimately finds it.
    import sys

    monkeypatch.setenv("PATH", "/nonexistent-dir-xyz")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    with pytest.raises(InputValidationError):
        resolve_colmap_bin("colmap")


def test_resolve_found_on_path(monkeypatch, tmp_path):
    fake = tmp_path / "colmap"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_colmap_bin("colmap") == str(fake)


def test_reconstruct_issues_feature_match_and_mapper(monkeypatch, tmp_path):
    """Regression: _reconstruct must run feature_extractor -> matcher -> MAPPER.
    (A dropped mapper call once produced valid matches but empty sparse/ silently.)"""
    import logging

    from video_to_3dgs.stages import run_colmap as rc

    calls: list[str] = []
    monkeypatch.setattr(rc, "_run_colmap", lambda args, log, cb="colmap": calls.append(args[0]))
    monkeypatch.setattr(rc.RunColmapStage, "_gpu_flag",
                        lambda self, *a, **k: "FeatureExtraction.use_gpu")

    class _C:
        colmap_bin = "colmap"; use_gpu = False; camera_model = "OPENCV"
        single_camera = True; sift_max_features = 8192; sequential_overlap = 10
        loop_detection = False; vocab_tree_path = None
        mapper_backend = "colmap"; glomap_bin = "glomap"

    rc.RunColmapStage()._reconstruct(
        None, tmp_path / "db.db", tmp_path / "img", tmp_path / "sparse", None,
        _C(), "sequential", logging.getLogger("t"), "colmap")
    assert calls == ["feature_extractor", "sequential_matcher", "mapper"], calls
