# Using COLMAP on this cluster

There is **no system/module COLMAP** on isiacluster (`module load colmap` does not
exist, and `colmap` is not on the default `PATH` on the login or GPU nodes).
COLMAP is instead provided **inside the `v2gs` conda env** (conda-forge build,
currently **COLMAP 3.13.0, compiled with CUDA**). This guide covers both how the
framework runs it for you and how to run it by hand.

```
/home/$USER/envs/v2gs/bin/colmap --version
# COLMAP 3.13.0 ... with CUDA
```

## 1. The easy path — let the framework run it
The `reconstruct` stage wraps the whole COLMAP flow (feature extraction →
matching → mapping → undistortion) on **node-local scratch** (COLMAP's SQLite DB
is hostile to NFS) and validates the result:

```bash
# on a GPU node (or inside an srun/sbatch allocation)
source scripts/_activate_env.sh          # puts the env's colmap on PATH, sets CUDA_HOME
python -m video_to_3dgs.cli reconstruct              --config configs/pipeline/scene_pavillon.yaml
python -m video_to_3dgs.cli validate-reconstruction  --config configs/pipeline/scene_pavillon.yaml
```

Relevant config knobs (`run_colmap:` in the pipeline YAML):

| key | meaning |
|---|---|
| `matcher` | `sequential` (video, default), `exhaustive`, `vocab_tree` |
| `camera_model` | e.g. `OPENCV` (default), `SIMPLE_RADIAL`, `PINHOLE` |
| `single_camera` | share one intrinsic across all frames |
| `use_gpu` | GPU SIFT (default `false` → robust CPU SIFT) |
| `sequential_overlap` | frames matched ahead/behind each frame |
| `use_masks` | pass object masks to feature extraction |
| `undistort` | produce undistorted images for training (default `true`) |
| `fallback_to_exhaustive` | retry with exhaustive matching if mapping fails |

Outputs land in `experiments/runs/<dataset_id>/colmap/`:
`database.db`, `sparse/0/{cameras,images,points3D}.bin`, undistorted `images/`,
`validation_report.json`, `trajectory.png`.

## 2. The manual path — run COLMAP yourself
Always work on **node-local scratch**, never `/home` (the SQLite DB thrashes NFS):

```bash
source scripts/_activate_env.sh
WORK=/var/tmp/$USER/colmap_manual; mkdir -p "$WORK"
IMAGES=/path/to/frames          # a directory of .jpg/.png frames
DB="$WORK/database.db"

# 1) features (CPU SIFT is robust everywhere; GPU SIFT needs a CUDA build + a display/EGL)
colmap feature_extractor \
    --database_path "$DB" --image_path "$IMAGES" \
    --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 0 --SiftExtraction.max_num_features 8192

# 2) matching (sequential for video; exhaustive for unordered sets / turntable)
colmap sequential_matcher --database_path "$DB" \
    --SequentialMatching.overlap 10 --FeatureMatching.use_gpu 0
# or: colmap exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 0

# 3) sparse mapping
mkdir -p "$WORK/sparse"
colmap mapper --database_path "$DB" --image_path "$IMAGES" --output_path "$WORK/sparse"
# -> "$WORK/sparse/0/{cameras,images,points3D}.bin"

# 4) undistort for downstream training
colmap image_undistorter --image_path "$IMAGES" \
    --input_path "$WORK/sparse/0" --output_path "$WORK/undist" --output_type COLMAP

# 5) inspect stats
colmap model_analyzer --path "$WORK/sparse/0"
```

Sync only the final `sparse/` + undistorted `images/` back to `/home`; leave the
DB on scratch.

> **COLMAP 3.13 option rename (important):** the GPU flags moved from
> `--SiftExtraction.use_gpu` / `--SiftMatching.use_gpu` to
> `--FeatureExtraction.use_gpu` / `--FeatureMatching.use_gpu` (but
> `--SiftExtraction.max_num_features` stayed). CPU-only COLMAP builds omit the
> `use_gpu` flags entirely. The framework auto-detects the accepted name per build.

## 3. GPU vs CPU SIFT
- **CPU SIFT** (`use_gpu 0`): slower but robust and headless-safe — the default.
- **GPU SIFT** (`use_gpu 1`): needs a CUDA-enabled COLMAP build **and** an OpenGL/EGL
  context; on headless nodes it often fails to create a display. Prefer CPU SIFT
  unless you have verified GPU SIFT works on your node.

## 4. When mapping fails ("mapper produced no model")
Usually too few feature matches between images. Try, in order:
1. **More/denser frames** — raise `extract_frames.target_fps` (more overlap).
2. **Higher resolution** — raise `extract_frames.resize_long_edge` (more features).
3. **Exhaustive matching** — `run_colmap.matcher: exhaustive` (no ordering assumption).
4. Check `colmap/validation_report.json` + `trajectory.png`, and the mapper output
   (the framework logs its last lines) for "No good initial image pair found".
5. Ensure the capture has **parallax** (translation, not pure rotation) and texture.

## 5. Installing a different COLMAP
COLMAP comes from `environment.yml` (conda-forge). To pin or change it, edit that
file and re-run `scripts/bootstrap_environment.sh`, or `mamba install -p
/home/$USER/envs/v2gs colmap=<version>`. GLOMAP (a faster global mapper) can be
added similarly and wired in as an alternative to the `mapper` step.
