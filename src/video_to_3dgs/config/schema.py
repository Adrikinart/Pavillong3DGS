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


class ObjectMaskCfg(_Base):
    """Per-view masks made by projecting a 3D volume, used to restrict the TRAINING loss.

    Distinct from ``generate_masks``, and deliberately so. That stage runs before SfM and
    feeds COLMAP, which is wrong for a subject like the Casque helmet: it has almost no
    stable features while the checkerboard beside it has many, so masking before SfM throws
    away the evidence that produces good poses. This stage runs *after* ``normalize_scene``,
    when camera poses exist, and its masks are used only by the loss.

    It also avoids segmentation entirely. ``rembg`` was measured on that capture and swung
    between 0.1 % and 45 % of the frame across the orbit (chrome, a wispy plume, a competing
    stand). Projecting a volume we already trust in 3D is deterministic and its errors are
    geometric rather than semantic.

    Why bother: with the loss restricted to the subject, the Gaussian budget stops mattering
    (flat from 190 k to 6 M on the Casque), which is a ~30x smaller model at no measurable
    cost. See docs/technique_transfer.md.
    """
    enabled: bool = False
    source: Literal["mesh", "box"] = "mesh"
    mesh_path: Optional[str] = None          # cropped object mesh, normalized frame
    box_center: Optional[list[float]] = None  # used when source == "box"
    box_half_extent: Optional[float] = None
    splat_px: int = 11        # mesh mode: radius drawn per projected vertex before closing
    dilate_px: int = 14       # slack; a tight silhouette clips thin structure (the plume)
    min_area_fraction: float = 0.0   # warn below this; 0 disables the check


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
    cap_max: int = 1_000_000            # default: ceiling on #gaussians; mcmc: target count
    noise_lr: float = 5e5               # mcmc only: SGLD noise scale


class BoundsCfg(_Base):
    """Constrain Gaussians to a volume.

    By default the volume is the auto AABB of (cameras + points) x (1+margin) -- a
    generous room bound that only culls runaway floaters. For an OBJECT capture where
    the subject occupies a small part of the scene (a helmet in a room), set an
    explicit tight box via ``box_center`` + ``box_half_extent`` to prune the
    environment and concentrate the Gaussian budget on the object. The centre is in
    the NORMALIZED frame (read it off the camera look-at of the reconstruction)."""
    enabled: bool = True
    margin: float = 0.5          # fraction of the scene extent added beyond the auto box
    prune_every: int = 500       # suppress out-of-box Gaussians this often
    box_center: Optional[list[float]] = None      # explicit box centre (normalized frame)
    box_half_extent: Optional[float] = None       # explicit box half-size (per axis)


class FloaterCfg(_Base):
    """Suppress 'flying' Gaussians (large, faint, or in empty space)."""
    enabled: bool = True
    max_scale_frac: float = 0.15  # prune Gaussians larger than this * scene extent
    min_opacity: float = 0.02     # hard-prune below this opacity at the end
    prune_every: int = 500


class NormalConsistencyCfg(_Base):
    """Depth-normal consistency: a self-supervised geometric signal (2DGS/GOF/PGSR
    use this internally). Penalises disagreement between the normal implied by each
    Gaussian's shortest axis and the normal implied by the rendered depth."""
    enabled: bool = False
    lambda_normal: float = 0.05
    start_iter: int = 7000        # let coarse geometry form before aligning it
    alpha_min: float = 0.5        # ignore pixels with little accumulated opacity


class DepthPriorCfg(_Base):
    """Monocular-depth regularization (sparse-view geometry prior)."""
    enabled: bool = False
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    lambda_depth: float = 0.1
    start_iter: int = 500
    loss: Literal["pearson", "l1_affine"] = "pearson"


class TrainCfg(_Base):
    backend: Literal["gsplat", "2dgs", "splatfacto", "orig_3dgs"] = "gsplat"
    bounds: "BoundsCfg" = Field(default_factory=lambda: BoundsCfg())
    floater: "FloaterCfg" = Field(default_factory=lambda: FloaterCfg())
    depth_prior: "DepthPriorCfg" = Field(default_factory=lambda: DepthPriorCfg())
    normal_consistency: "NormalConsistencyCfg" = Field(default_factory=lambda: NormalConsistencyCfg())
    train_run_id: Optional[str] = None  # auto if None
    seed: int = 42
    max_iterations: int = 30000
    image_downscale: int = 1
    sh_degree: int = 3
    ssim_lambda: float = 0.2
    l1_lambda: float = 0.8
    use_masks: bool = True
    mixed_precision: bool = True
    pose_optimization: bool = False   # learn per-training-camera SE(3) deltas
    pose_lr: float = 1.0e-5           # small: poses should refine, not wander
    # Per-image appearance latents (GLO / NeRF-W style) decoded to a global affine
    # colour transform. Enable when merging clips with differing auto-exposure/WB.
    appearance_embedding: bool = False
    appearance_dim: int = 16
    appearance_lr: float = 1e-3
    # 'affine': one 3x3+bias per image (cannot encode geometry at all).
    # 'bilateral': coarse (x,y,luma) grid of affine transforms - can represent
    # vignetting/spatially varying response, at the cost of a weaker capacity bound.
    appearance_model: Literal["affine", "bilateral"] = "affine"
    bilateral_grid_wh: int = 16
    bilateral_grid_luma: int = 8
    bilateral_tv_lambda: float = 10.0
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
    orbit_arc_deg: float = 80.0          # front-facing sweep (single-sided captures only)
    # For captures measured to be true orbits the fly-around radius is the capture's own
    # median camera distance times this scale, so novel views reproduce the framing the
    # photographer chose. Only used on the single-sided path via framing_margin below.
    orbit_radius_scale: float = 1.2
    framing_margin: float = 1.25         # single-sided: pull-back so the object fits
    orbit_width: int = 960
    orbit_height: int = 540
    video_fps: int = 30
    progression_fps: int = 4             # overview progression is a few checkpoints
    progression_hold: int = 6            # frames to hold each checkpoint
    crop_to_object: bool = True          # render only Gaussians in the object box (kill floaters)
    # Half-size of that box for measured-orbit captures, as a fraction of the capture's
    # median camera distance. The stored point-cloud box spans the whole room for an
    # object filmed indoors, so it removes nothing; this one is centred on the subject.
    # Raise it if the fly-around clips scenery you want, lower it if haze survives.
    object_crop_scale: float = 0.5


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
    object_masks: ObjectMaskCfg = Field(default_factory=ObjectMaskCfg)
    split_dataset: SplitCfg = Field(default_factory=SplitCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    evaluate: EvalCfg = Field(default_factory=EvalCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    report: ReportCfg = Field(default_factory=ReportCfg)
    monitoring: MonitoringCfg = Field(default_factory=MonitoringCfg)

    schema_version: int = 1
