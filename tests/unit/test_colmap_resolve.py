"""colmap binary resolution (regression for the Slurm PATH pitfall)."""

import pytest

from video_to_3dgs.core.errors import InputValidationError
from video_to_3dgs.stages.run_colmap import resolve_colmap_bin


def test_resolve_explicit_existing_path():
    # an existing absolute path is honored as-is
    assert resolve_colmap_bin("/bin/sh") == "/bin/sh"


def test_resolve_raises_when_absent(monkeypatch):
    # ensure 'colmap' is not found on PATH nor in sys.prefix/bin
    monkeypatch.setenv("PATH", "/nonexistent-dir-xyz")
    with pytest.raises(InputValidationError):
        resolve_colmap_bin("colmap")


def test_resolve_found_on_path(monkeypatch, tmp_path):
    fake = tmp_path / "colmap"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_colmap_bin("colmap") == str(fake)
