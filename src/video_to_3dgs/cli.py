"""Unified CLI for the video_to_3dgs pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config.loader import (
    derive_dataset_id,
    freeze_config,
    load_config,
    load_frozen_config,
    make_layout,
)
from .core.atomicio import sha256_str
from .core.errors import V2GSError
from .core.logging import configure_logging, get_logger
from .core.manifest import Manifest
from .core.paths import RunLayout
from .core.runner import StageRunner
from .core.stage import StageContext

# stage registry (import lazily-safe: these modules avoid importing torch at module load)
from .stages.inspect_video import InspectVideoStage
from .stages.extract_frames import ExtractFramesStage
from .stages.filter_frames import FilterFramesStage
from .stages.generate_masks import GenerateMasksStage
from .stages.run_colmap import RunColmapStage
from .stages.validate_colmap import ValidateColmapStage
from .stages.normalize_scene import NormalizeSceneStage
from .stages.split_dataset import SplitDatasetStage
from .stages.train import TrainStage
from .stages.evaluate import EvaluateStage
from .stages.export import ExportStage

REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE_CLASSES = [
    InspectVideoStage, ExtractFramesStage, FilterFramesStage, GenerateMasksStage,
    RunColmapStage, ValidateColmapStage, NormalizeSceneStage, SplitDatasetStage,
    TrainStage, EvaluateStage, ExportStage,
]
STAGE_BY_NAME = {c.name: c for c in STAGE_CLASSES}

# CLI command name -> stage name (for the per-stage subcommands)
COMMAND_TO_STAGE = {
    "inspect-video": "inspect_video",
    "extract-frames": "extract_frames",
    "filter-frames": "filter_frames",
    "generate-masks": "generate_masks",
    "reconstruct": "run_colmap",
    "validate-reconstruction": "validate_colmap",
    "normalize": "normalize_scene",
    "split": "split_dataset",
    "train": "train",
    "evaluate": "evaluate",
    "export": "export",
}


def _quick_signature(cfg) -> str:
    parts = []
    for v in cfg.videos:
        p = Path(v)
        try:
            parts.append(f"{p.name}:{p.stat().st_size}")
        except OSError:
            parts.append(p.name)
    return sha256_str("|".join(parts))[7:15]  # strip 'sha256:' prefix, take 8 chars


def _build_context(args, *, frozen: bool = True) -> tuple[StageContext, RunLayout]:
    log = get_logger("cli")
    overrides = list(getattr(args, "set", []) or [])
    if frozen and getattr(args, "dataset_id", None):
        layout = RunLayout(runs_root=_runs_root(args), dataset_id=args.dataset_id)
        cfg = load_frozen_config(layout)
    else:
        cfg = load_config(getattr(args, "config", None), profile=getattr(args, "profile", None),
                          overrides=overrides)
        dataset_id = (getattr(args, "dataset_id", None) or cfg.dataset_id
                      or derive_dataset_id(cfg, _quick_signature(cfg)))
        layout = make_layout(cfg, REPO_ROOT, dataset_id)
        if frozen and layout.config_resolved.exists():
            cfg = load_frozen_config(layout)

    layout.ensure_base_dirs()
    configure_logging(getattr(args, "verbose", False),
                      jsonl_path=layout.logs_dir / "cli.jsonl")
    manifest = Manifest(layout.manifest)
    params = {}
    if getattr(args, "train_run_id", None):
        params["train_run_id"] = args.train_run_id
    if getattr(args, "videos", None):
        params["videos"] = args.videos
    ctx = StageContext(layout=layout, config=cfg, manifest=manifest, logger=log,
                       repo_root=REPO_ROOT, dry_run=getattr(args, "dry_run", False),
                       force=getattr(args, "force", False),
                       verbose=getattr(args, "verbose", False), params=params)
    return ctx, layout


def _runs_root(args) -> Path:
    # best-effort: use default runs root under repo
    return REPO_ROOT / "experiments" / "runs"


def _prepare(ctx: StageContext) -> None:
    """Initialize a run: freeze config, capture provenance, init manifest header."""
    from .core.provenance import software_block

    freeze_config(ctx.config, ctx.layout, force=ctx.force)
    software, freeze_txt = software_block(ctx.repo_root)
    (ctx.layout.logs_dir / "pip_freeze.txt").write_text(freeze_txt or "")
    ctx.manifest.init_header(dataset_id=ctx.layout.dataset_id,
                             videos=[{"path": v} for v in ctx.config.videos],
                             software=software)
    ctx.logger.info("prepared run '%s' at %s", ctx.layout.dataset_id, ctx.layout.run_dir)


# --------------------------------------------------------------------------- #
def cmd_inspect_env(args) -> int:
    from . import environment
    configure_logging(args.verbose)
    report = environment.inspect_environment(gpu_check=args.gpu_check, repo_root=REPO_ROOT)
    print(environment.summarize(report))
    if args.out:
        environment.write_report(report, args.out)
    ok = True
    if args.gpu_check:
        ok = (report.get("cuda_smoke", {}).get("forward") is True
              and report.get("cuda_smoke", {}).get("backward") is True)
    return 0 if ok else 7


def cmd_prepare(args) -> int:
    ctx, _ = _build_context(args, frozen=False)
    _prepare(ctx)
    return 0


def cmd_stage(args, stage_name: str) -> int:
    ctx, layout = _build_context(args, frozen=True)
    if not layout.config_resolved.exists():
        _prepare(ctx)
    runner = StageRunner(ctx)
    stage = STAGE_BY_NAME[stage_name]()
    state = runner.execute(stage)
    ctx.logger.info("stage %s -> %s", stage_name, state.value)
    return 0


def cmd_run_all(args) -> int:
    ctx, layout = _build_context(args, frozen=False)
    _prepare(ctx)
    # reload frozen for consistency
    ctx.config = load_frozen_config(layout)
    runner = StageRunner(ctx)
    stages = [c() for c in STAGE_CLASSES]
    # masking is opt-in: drop from run-all if disabled to avoid a no-op required output
    results = runner.run_all(stages, from_stage=args.from_stage, to_stage=args.to_stage,
                             only=args.only)
    ctx.logger.info("run-all results: %s", results)
    failed = [k for k, v in results.items() if v == "FAILED"]
    return 1 if failed else 0


def cmd_status(args) -> int:
    from .core.status import read_status

    ctx, layout = _build_context(args, frozen=True)
    print(f"dataset: {layout.dataset_id}  ({layout.run_dir})")
    for c in STAGE_CLASSES:
        rec = read_status(layout.status_file(c.name), c.name)
        extra = ""
        if rec.metrics:
            extra = "  " + " ".join(f"{k}={v}" for k, v in list(rec.metrics.items())[:3])
        print(f"  {c.name:22s} {rec.state:10s}{extra}")
    return 0


def cmd_report(args) -> int:
    ctx, layout = _build_context(args, frozen=True)
    tr = args.train_run_id
    if tr is None:
        tdir = layout.trainings_dir
        trs = sorted([p.name for p in tdir.iterdir()]) if tdir.exists() else []
        if not trs:
            print("no trainings found")
            return 1
        tr = trs[-1]
    ej = layout.eval_json(tr)
    if not ej.exists():
        print(f"no eval.json for {tr}; run evaluate first")
        return 1
    import json

    from .monitoring.report import write_report
    write_report(ctx, tr, json.loads(ej.read_text()))
    print(f"report: {layout.report_dir(tr) / 'report.md'}")
    return 0


# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser, *, stage: bool = True) -> None:
    p.add_argument("--config", type=str, help="user pipeline YAML")
    p.add_argument("--profile", type=str, help="cluster profile name")
    p.add_argument("--dataset-id", type=str, help="explicit dataset id (existing run)")
    p.add_argument("--set", action="append", default=[], help="override key=value (repeatable)")
    p.add_argument("--verbose", action="store_true")
    if stage:
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--force", action="store_true")
        p.add_argument("--resume", action="store_true", help="(default) skip completed stages")
        p.add_argument("--train-run-id", type=str, help="training run id (train/evaluate/export)")
        p.add_argument("--videos", nargs="*", help="override source videos")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-to-3dgs",
                                     description="Video -> 3D Gaussian Splatting pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("inspect-env", help="inspect cluster/CUDA/GPU environment")
    pe.add_argument("--gpu-check", action="store_true", help="run CUDA + gsplat smoke tests")
    pe.add_argument("--out", type=str, help="write JSON report to this path")
    pe.add_argument("--verbose", action="store_true")
    pe.set_defaults(func=cmd_inspect_env)

    pp = sub.add_parser("prepare", help="initialize a run (freeze config, provenance)")
    _add_common(pp)
    pp.set_defaults(func=cmd_prepare)

    for cmd, stage_name in COMMAND_TO_STAGE.items():
        sp = sub.add_parser(cmd, help=f"run stage: {stage_name}")
        _add_common(sp)
        sp.set_defaults(func=lambda a, s=stage_name: cmd_stage(a, s))

    ra = sub.add_parser("run-all", help="run the full pipeline")
    _add_common(ra)
    ra.add_argument("--from-stage", type=str)
    ra.add_argument("--to-stage", type=str)
    ra.add_argument("--only", type=str)
    ra.set_defaults(func=cmd_run_all)

    st = sub.add_parser("status", help="show per-stage status")
    _add_common(st, stage=False)
    st.set_defaults(func=cmd_status)

    rp = sub.add_parser("report", help="(re)generate the run report")
    _add_common(rp, stage=False)
    rp.add_argument("--train-run-id", type=str)
    rp.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except V2GSError as e:
        get_logger("cli").error("%s: %s", type(e).__name__, e)
        return e.exit_code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
