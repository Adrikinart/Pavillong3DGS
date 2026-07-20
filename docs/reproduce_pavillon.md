# Reproducing the Pavillon 3DGS model

End-to-end recipe to re-create the carved-panel ("Pavillon") 3D Gaussian Splatting
reconstruction from the raw video, including the **A3 geometry regularizers**
(room bounds + anti-floater + monocular depth prior).

Everything below is driven by two config files; no code edits are needed.

| Model | Config | Test PSNR / SSIM / LPIPS | Gaussians | Pruned | `.ply` |
|---|---|---|---|---|---|
| Baseline (GLOMAP + 3DGS) | `pavillon_orbit_hq.yaml` | **23.9** / 0.825 / 0.245 | 1.53 M | — | 374 MB |
| **A3 regularized** | `pavillon_orbit_reg.yaml` | 22.3 / 0.796 / 0.290 | **1.22 M** | 20.2 % | **301 MB** |
| A3 ablation (depth off) | `pavillon_orbit_reg_nodepth.yaml` | 22.6 / 0.798 / 0.294 | 1.23 M | 18.0 % | 306 MB |
| **High-detail** (2560px, 282 imgs) | `pavillon_orbit_hidetail.yaml` | 23.1 / **0.846** / 0.310 | 1.24 M | 18.0 % | 309 MB |
| Multi-clip + appearance | `pavillon_multiclip.yaml` | 22.0 / 0.833 / 0.342 | 1.23 M | 18.9 % | 304 MB |

> **The high-detail row is NOT directly comparable to the rows above it.** It is a
> different dataset (its own `object_name` → its own COLMAP, splits and test views:
> **28 test views at 2560px** vs 18 at 1600px). Cross-resolution PSNR/LPIPS
> comparisons are unreliable. Compare it against the A3 model on *distribution*
> instead, where it wins on the harder split:
>
> | | A3 @1600px (18 views) | High-detail @2560px (28 views) |
> |---|---|---|
> | median PSNR | 23.43 | **24.48** |
> | mean SSIM | 0.796 | **0.846** |
> | best view | 26.5 | **31.9** |
> | catastrophic views (<18 dB) | 3/18 (17 %) | 3/28 (**11 %**) |
>
> **Multi-clip underperforms — `pavillon_orbit_hidetail` is still recommended — but
> most of the apparent gap was a measurement bug.** `evaluate` used to render without
> restoring the appearance model, scoring views on the uncorrected output while
> training optimised corrected ones. Fixing that (per-clip appearance, see
> `AppearanceModel.canonical_for`) moved multi-clip from 19.0 to **22.0 PSNR**,
> cutting the gap from 4.06 dB to **1.05 dB**. **If you evaluate a run trained with
> `appearance_embedding`, make sure this correction is applied — otherwise the model
> is scored unfairly.**
>
> A modest regression does remain (median 24.48 → 22.82; catastrophic views 11 % →
> 24 %). The per-clip split localises it: the *added* clip's views score fine
> (img_9649 mean 23.68) while the *original* clip's degrade (img_9647 21.81) — the
> 1.5 M Gaussian budget now covers a larger volume. The untested fix is to raise
> `densification.cap_max` with the volume; the failure is budget dilution, not
> appearance and not SfM.
>
> SfM also improved outright: **282/282 images registered (100 %)** with **152 792**
> sparse points, versus 181/193 (94 %) and 80 913 at 1600px.

The A3 model trades ~1.4 dB PSNR for markedly cleaner geometry: **18.2 % → 0 %**
near-transparent "haze" Gaussians, max Gaussian scale **1.00 → 0.16**, median
opacity **0.51 → 0.88**. See [Step 6](#6-verify-the-floater-reduction).

**Where that PSNR actually goes (measured, not assumed).** The depth-off ablation
recovers only **0.26 dB**, so the depth prior is *not* the cost — the **final hard
prune** is. `floater.min_opacity: 0.02` removes faint Gaussians that still
contribute to held-out views. Validation PSNR is ~24.5 for both regularized runs,
on par with the baseline; the gap appears only on the held-out test split.

> **Tuning knob:** to keep most of the floater cleanup at a smaller photometric
> cost, lower `floater.min_opacity` toward `0.005` (the densifier's own prune
> threshold). Raise it for a leaner, cleaner `.ply`. Do **not** disable the depth
> prior for PSNR — it is nearly free and slightly *increases* floater removal.

---

## 0. Hardware / driver prerequisites (read this first)

The environment pins **PyTorch cu128/cu130**, which requires a *recent* NVIDIA
driver. On this cluster that means **only the Blackwell nodes work**:

| Node | GPU | Usable? (verified by probe) |
|---|---|---|
| GPURACK2 | RTX PRO 4500 Blackwell (34 GB) | ✅ **verified** — SM 12.0, torch 2.13.0+cu130 |
| GPURACK5 | RTX PRO 6000 (96 GB) | ✅ ran all earlier work — but currently `DOWN` |
| GPURACK4 | RTX 5080 (16 GB) | ❌ driver 12080 too old (despite being Blackwell) |
| GPURACK1 / GPURACK3 | 3090 / 4090 | ❌ driver 12080 too old |

Do **not** assume a node is usable because its GPU is new — GPURACK4 has a Blackwell
5080 yet still reports `CUDA initialization: driver too old (found version 12080)`
and `torch.cuda.is_available() == False`. Probe before submitting:

```bash
srun --partition=<p> --nodelist=<node> --gres=gpu:1 --time=00:03:00 \
  bash -lc 'source scripts/_activate_env.sh >/dev/null 2>&1;
            python -c "import torch; print(torch.cuda.is_available())"'
```

Note that CPU-only stages (frame extraction, filtering, **COLMAP/GLOMAP** — SIFT
runs on CPU by default) work fine on the old-driver nodes; only `train`, `evaluate`
and the renders need CUDA. So it is legitimate to reconstruct on one node and then
re-run just `--from-stage train` on a CUDA-capable one.

> **Memory warning.** Slurm here reports `RealMemory=1` and gathers no memory
> accounting (`SelectTypeParameters=CR_CORE`), so **it enforces no RAM limit**. A
> runaway job can exhaust node RAM until `slurmd` stops responding and the node is
> marked `DOWN+NOT_RESPONDING`. Always cap the CUDA build parallelism with
> `MAX_JOBS` (see Step 1) — the gsplat JIT build is the only unbounded-RAM step.

## 1. Environment

```bash
scripts/bootstrap_environment.sh          # builds /home/$USER/envs/v2gs (one-time)
```

The depth prior additionally needs `transformers` and the DepthAnything-v2
weights. Install/fetch these **on the login node** (compute nodes may lack
internet; the HF cache on NFS `~/.cache/huggingface` is then readable by them):

```bash
/home/$USER/envs/v2gs/bin/pip install "transformers>=4.45"
/home/$USER/envs/v2gs/bin/python -c "
from transformers import pipeline
pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')"
```

**Pre-flight check on the target node** (also pre-builds and caches the gsplat
CUDA extension, so the long training never pays the RAM-heavy compile):

```bash
srun --partition=rtxpro --nodelist=GPURACK2 --gres=gpu:1 --cpus-per-task=4 \
     --time=00:25:00 bash -lc \
  'cd "$SLURM_SUBMIT_DIR"; export MAX_JOBS=4; source scripts/_activate_env.sh; \
   python scripts/gsplat_selftest.py'
```

Expect `GSPLAT_OK=True`. The build takes ~2 min and ~3 GB RAM at `MAX_JOBS=4`.
`TORCH_EXTENSIONS_DIR` is **node-local**, so repeat this once per node.

## 2. Data

One 4K iPhone clip of the carved ceiling panel, stood vertical:

```
/home/AdrienK/Datasets/VideosForNVSpersonnal/Pavillon/IMG_9647.MOV
```

Point `videos:` in the config at your copy. **Only this single orbit is used.**
Merging clips was originally blocked by differing auto-exposure/white balance,
which `train.appearance_embedding` now handles — but merging IMG_9649 was measured
and still produced a worse model (see the negative result above). Exposure was not
the only obstacle: the second clip enlarges the reconstructed volume while adding
few views, so the Gaussian budget is spread thinner. Merge clips only when they
re-observe the *same* surfaces from new angles, and raise `densification.cap_max`
if the merged volume grows.

IMG_9648 is unusable in any configuration — it is a weakly-connected walking clip
that does not reconstruct.

## 3. Run the reconstruction

**Fresh reproduction** (runs every stage: extract → filter → COLMAP/GLOMAP →
normalize → split → train → evaluate → export → visualize):

```bash
# A3 regularized model (the deliverable)
sbatch --partition=rtxpro --nodelist=GPURACK2 --gres=gpu:1 --cpus-per-task=8 \
       --export=ALL,MAX_JOBS=4,OMP_NUM_THREADS=4 \
       scripts/slurm/train.sbatch configs/pipeline/pavillon_orbit_reg.yaml

# plain baseline, for comparison
sbatch scripts/slurm/train.sbatch configs/pipeline/pavillon_orbit_hq.yaml
```

**Adding a second training to an existing dataset dir.** The resolved config is
frozen per dataset directory and is authoritative, so an edited config is ignored
unless you re-freeze it. To train a *sibling* run reusing the same COLMAP:

```bash
sbatch ... scripts/slurm/train.sbatch configs/pipeline/pavillon_orbit_reg.yaml \
       --force --from-stage train
```

`--force` re-freezes the config; `--from-stage train` skips (and therefore never
re-runs) the expensive GLOMAP reconstruction. Each config sets a distinct
`train.train_run_id`, so runs never overwrite each other.

Runtime on an RTX PRO 6000: ~9 min for 30 k iterations, plus GLOMAP.

## 4. What the configs do

Both use **GLOMAP** as the mapper: on this low-overlap capture it registers
**181/193** images versus **82** for incremental COLMAP — a 2.2× coverage gain
that is the single biggest quality lever.

`pavillon_orbit_reg.yaml` additionally enables the three A3 regularizers:

```yaml
train:
  bounds:                    # keep Gaussians inside the captured room
    enabled: true
    margin: 0.5              # AABB(cameras ∪ points) expanded by 50 %
    prune_every: 500
  floater:                   # kill "flying" Gaussians
    enabled: true
    max_scale_frac: 0.15     # suppress Gaussians > 0.15 · scene_extent
    min_opacity: 0.02        # final hard prune below this opacity
    prune_every: 500
  depth_prior:               # sparse-view geometry signal
    enabled: true
    model: depth-anything/Depth-Anything-V2-Small-hf
    lambda_depth: 0.1
    start_iter: 2000         # let photometric settle first
    loss: pearson            # scale/shift-invariant
```

- **Room bounds** — Gaussians outside the box are opacity-suppressed each
  `prune_every` steps, then removed by the densifier's opacity prune.
- **Anti-floater** — oversized Gaussians are suppressed during training; on clean
  completion a **final hard prune** drops everything below `min_opacity` or
  outside the box, yielding a floater-free `.ply`.
- **Depth prior** — DepthAnything-v2 depth is cached once to
  `experiments/runs/<dataset_id>/depth_prior/` and compared against the rendered
  expected depth (`RGB+ED`) with a Pearson (scale/shift-invariant) loss, so the
  metric-less monocular prior constrains only depth *ordering*.

Set `depth_prior.enabled: false` (i.e. `pavillon_orbit_reg_nodepth.yaml`) to keep
the floater cleanup without the depth prior's photometric cost.

## 5. Outputs

Under `experiments/runs/pavillon_orbit_<sha8>/`:

```
trainings/<train_run_id>/
  checkpoints/ckpt_*.pt          # final one is post-prune
  renders/val_*/                 # per-validation-interval renders
  report/figures/*.png           # curves, qualitative, stats, centers, per-view
  videos/orbit.mp4               # front-arc sweep framing the whole object
  videos/training_progression.mp4
  eval.json                      # held-out metrics
exports/<train_run_id>/
  point_cloud.ply                # standard INRIA/gsplat 3DGS .ply
  cameras.json, normalize_transform.json, COORDINATES.md
```

`experiments/registry.csv` gains one row per evaluated run (metrics, #Gaussians,
git SHA, config path) — the index for comparing experiments.

## 6. Verify the floater reduction

Compare any two exported models **without a GPU** (parses the `.ply`s directly):

```bash
python scripts/floater_spatial.py pavillon_orbit_<sha8> \
  experiments/runs/.../exports/gsplat_run/point_cloud.ply \
  experiments/runs/.../exports/gsplat_reg_30k/point_cloud.ply \
  floater_spatial.png
```

It reports the fraction of Gaussians outside the room box and plots the centers
from two angles. On this capture the baseline shows a diffuse streak of Gaussians
spraying toward the cameras; the regularized model is a tight, clean cluster.

## 7. Gotchas

- **Judge a run by looking at the renders, not just PSNR.** Check
  `trainings/<id>/renders/val_*/` a couple of thousand iterations in — a dead end
  (flat/blurry/floaters) is obvious in seconds. This is how the 2DGS backend was
  ruled out for this capture (flat beige renders at PSNR ~13).
- **2DGS is a dead end here** — it needs multi-view surface coverage this
  single-side capture lacks. Kept as an experimental backend only.
- **Never run COLMAP/SQLite on `/home`** — stages already stage to node-local
  `/var/tmp/$USER`; keep it that way.
- **Never `git add -A`** in this repo (NFS + large artifacts). Stage explicit paths.
- If training reports `CUDA not available`, you are on an old-driver node — see
  the table in Step 0.
