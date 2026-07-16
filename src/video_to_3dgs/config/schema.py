"""Typed configuration schema (pydantic v2).

Layered YAML is merged, validated into ``PipelineConfig``, then frozen to
``config_resolved.yaml`` per run. Backend-independent knobs live in typed models;
backend-specific knobs ride in ``TrainCfg.backend_opts`` (opaque dict) so the
schema does not grow a field per backend.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    # frozen so a resolved config is fully immutable (nested models included),
    # which keeps stage fingerprints stable within a run.
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Storage / cluster
# --------------------------------------------------------------------------- #
class StorageCfg(_Base):
    runs_root: str = "experiments/runs"
    raw_data_root: str = "data/raw"
    processed_data_root: str = "data/processed"
    scratch_root: str = "auto"          # auto -> resolve_scratch_root()
    cache_root: str = "auto"            # for TORCH_EXTENSIONS_DIR / pip cache


class ClusterProfile(_Base):
    name: str = "local"
    scheduler: Literal["auto", "slurm", "local"] = "auto"
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    nodelist: Optional[str] = None
    gpu_type: Optional[str] = None      # e.g. "RTX_PRO_6000"
    cuda_arch: Optional[str] = None     # e.g. "12.0" (sm_120)
    torch_cuda: Optional[str] = None    # e.g. "cu128"
    gpus: int = 1
    cpus_per_task: int = 8
    memory_gb: int = 32
    time_limit: str = "24:00:00"
    signal_before_timeout_seconds: int = 120
    requeue: bool = True
    env_prefix: str = "/home/${USER}/envs/v2gs"


# --------------------------------------------------------------------------- #
# Stage configs
# --------------------------------------------------------------------------- #
class InspectVideoCfg(_Base):
    min_duration_s: float = 0.5


class ExtractFramesCfg(_Base):
    strategy: Literal["fixed_fps", "fixed_interval", "max_frames", "keyframe",
                      "manual"] = "fixed_fps"
    target_fps: float = 2.0
    frame_interval: int = 15            # for fixed_interval
    max_frames: int = 400
    min_frames: int = 40
    manual_frames: list[int] = Field(default_factory=list)
    preserve_original_resolution: bool = False
    resize_long_edge: Optional[int] = 1600   # None = keep original
    output_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = 95
    honor_rotation: bool = True         # apply iPhone displaymatrix rotation


class FilterFramesCfg(_Base):
    enabled: bool = True
    blur_var_min: float = 60.0          # variance-of-Laplacian threshold
    luminance_min: float = 15.0
    luminance_max: float = 245.0
    saturated_frac_max: float = 0.35
    underexposed_frac_max: float = 0.60
    duplicate_phash_dist_min: int = 4   # min hamming distance to previous kept frame
    keep_fraction: Optional[float] = None    # cap on fraction kept (None = no cap)
    min_kept: int = 30                  # coverage safeguard: never go below this
    make_contact_sheets: bool = True


class MaskCfg(_Base):
    enabled: bool = False
    backend: Literal["rembg", "sam2", "imported", "none"] = "rembg"
    model: str = "isnet-general-use"
    imported_dir: Optional[str] = None
    dilate_px: int = 4
    minimum_area_fraction: float = 0.02
    maximum_area_fraction: float = 0.95
    preserve_pedestal: bool = False
    temporal_propagation: bool = False
    manual_review: bool = False


class ColmapCfg(_Base):
    matcher: Literal["sequential", "exhaustive", "vocab_tree"] = "sequential"
    # mapper backend: colmap (incremental) or glomap (global; more robust on
    # low-overlap / hard sets). glomap reuses COLMAP's features+matches.
    mapper_backend: Literal["colmap", "glomap"] = "colmap"
    glomap_bin: str = "glomap"
    camera_model: str = "OPENCV"
    single_camera: bool = True
    use_gpu: bool = False               # GPU SIFT on sm_120 unverified; default CPU
    sift_max_features: int = 8192
    sequential_overlap: int = 10
    loop_detection: bool = True
    vocab_tree_path: Optional[str] = None
    use_masks: bool = True              # pass masks to feature extraction if present
    undistort: bool = True
    fallback_to_exhaustive: bool = True
    colmap_bin: str = "colmap"


class ColmapValidationCfg(_Base):
    minimum_registration_ratio: float = 0.70
    maximum_mean_reprojection_error_px: float = 2.0
    minimum_sparse_points: int = 1000
    allow_multiple_models: bool = False
    hard_fail: bool = False             # False -> warn + continue; True -> raise


class NormalizeCfg(_Base):
    method: Literal["cameras", "points", "pca"] = "cameras"
    up_axis: Optional[Literal["x", "y", "z", "-x", "-y", "-z"]] = None  # None=estimate
    scale_percentile: float = 95.0
    outlier_std: float = 3.0
    exclude_pedestal: bool = False


class SplitCfg(_Base):
    strategy: Literal["pose_aware", "periodic", "random"] = "pose_aware"
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    holdout_every: int = 8              # periodic: hold out every Nth frame
    minimum_translation_distance: float = 0.0
    minimum_rotation_distance_deg: float = 0.0
    seed: int = 42


class DensifyCfg(_Base):
    strategy: Literal["default", "mcmc"] = "default"
    start_iteration: int = 500
    stop_iteration: int = 15000
    interval: int = 100
    grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3000
    prune_opacity: float = 0.005
    cap_max: int = 1_000_000            # ceiling on #gaussians (health)


class TrainCfg(_Base):
    backend: Literal["gsplat", "2dgs", "splatfacto", "orig_3dgs"] = "gsplat"
    train_run_id: Optional[str] = None  # auto if None
    seed: int = 42
    max_iterations: int = 30000
    image_downscale: int = 1
    sh_degree: int = 3
    ssim_lambda: float = 0.2
    l1_lambda: float = 0.8
    use_masks: bool = True
    mixed_precision: bool = True
    pose_optimization: bool = False
    appearance_embedding: bool = False
    antialiasing: bool = True
    lr_means: float = 1.6e-4
    checkpoint_interval: int = 5000
    validation_interval: int = 2000
    val_render_count: int = 4
    max_val_images: int = 8
    early_stop_patience: int = 0        # 0 = disabled
    densification: DensifyCfg = Field(default_factory=DensifyCfg)
    backend_opts: dict[str, Any] = Field(default_factory=dict)


class EvalCfg(_Base):
    splits: list[str] = Field(default_factory=lambda: ["test"])
    compute_lpips: bool = True
    masked_metrics: bool = True
    make_orbit_video: bool = True
    orbit_frames: int = 120
    worst_best_k: int = 4


class ExportCfg(_Base):
    formats: list[Literal["ply", "cameras", "transforms"]] = Field(
        default_factory=lambda: ["ply", "cameras", "transforms"]
    )
    ply_include_sh: bool = True


class MonitoringCfg(_Base):
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: Optional[str] = None
    gpu_sample_interval_s: float = 15.0


class ReportCfg(_Base):
    """Figures + videos generated post-training by the `visualize` stage."""
    enabled: bool = True
    # figures: training_curves(F1) qualitative(F2) gaussian_stats+per_view(F3) gaussian_centers(F4)
    figures: list[Literal["training_curves", "qualitative", "gaussian_stats",
                          "per_view", "gaussian_centers"]] = Field(
        default_factory=lambda: ["training_curves", "qualitative", "gaussian_stats",
                                 "per_view", "gaussian_centers"])
    # videos: orbit(V1) progression(V3)
    videos: list[Literal["orbit", "progression"]] = Field(
        default_factory=lambda: ["orbit", "progression"])
    orbit_frames: int = 120
    orbit_elevation_deg: float = 12.0
    orbit_arc_deg: float = 80.0          # front-facing sweep (not full 360 for single-side)
    orbit_radius_scale: float = 1.2      # (legacy; framing_margin now controls distance)
    framing_margin: float = 1.25         # pull-back so the whole object fits in view
    orbit_width: int = 960
    orbit_height: int = 540
    video_fps: int = 30
    progression_fps: int = 4             # overview progression is a few checkpoints
    progression_hold: int = 6            # frames to hold each checkpoint
    crop_to_object: bool = True          # render only Gaussians in the object box (kill floaters)


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
class PipelineConfig(_Base):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Optional[str] = None    # derived from video bytes if None
    object_name: Optional[str] = None
    capture_mode: Literal["orbit", "turntable"] = "orbit"
    videos: list[str] = Field(default_factory=list)   # source video paths

    storage: StorageCfg = Field(default_factory=StorageCfg)
    profile: ClusterProfile = Field(default_factory=ClusterProfile)

    inspect_video: InspectVideoCfg = Field(default_factory=InspectVideoCfg)
    extract_frames: ExtractFramesCfg = Field(default_factory=ExtractFramesCfg)
    filter_frames: FilterFramesCfg = Field(default_factory=FilterFramesCfg)
    generate_masks: MaskCfg = Field(default_factory=MaskCfg)
    run_colmap: ColmapCfg = Field(default_factory=ColmapCfg)
    validate_colmap: ColmapValidationCfg = Field(default_factory=ColmapValidationCfg)
    normalize_scene: NormalizeCfg = Field(default_factory=NormalizeCfg)
    split_dataset: SplitCfg = Field(default_factory=SplitCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    evaluate: EvalCfg = Field(default_factory=EvalCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    report: ReportCfg = Field(default_factory=ReportCfg)
    monitoring: MonitoringCfg = Field(default_factory=MonitoringCfg)

    schema_version: int = 1
