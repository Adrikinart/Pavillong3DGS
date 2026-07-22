# Technique transfer: Pavillon → Casque

Every technique developed on the Pavillon (single-sided carved wall relief), assessed
for the Casque (orbit of a free-standing reflective helmet), with the outcome. The
point is that a result is not a law: several Pavillon findings **invert** on a capture
with the opposite geometry, and the table records which.

Legend: ✅ applied & helps · ➖ applied, low impact here · 🔬 under test · ❌ tested,
does not help · ⏭️ deliberately skipped.

| Technique | Pavillon result | Casque relevance | Status |
|---|---|---|---|
| **GLOMAP** global SfM | 181/193 vs 82 (incremental) | High — 4 clips, two cameras | ✅ 533/536 @ 1.57 px |
| Scene-scale + LR-decay fixes | PSNR 14 → 24 | Framework-wide | ✅ automatic |
| **Resolution** (no downscale) | +1.8 dB, tail → 0 | Limited — only 1 of 4 clips is 4K | ➖ 2560 px, capped by 1080p clips |
| **Anti-floater** (bounds + prune) | 18.2 % → 0 % haze | Yes | ✅ applied |
| **Depth prior** (Pearson) | nearly free (0.26 dB) | Lower — an orbit already has parallax | ➖ on, low impact |
| **Capacity** sweep | 375 k optimal (*less is more*) | **INVERTS** — an orbit is not sparse-view | ✅ **1.5 M** optimal (4× the Pavillon) |
| **MCMC** densification | parity, cleaner control | Yes | ✅ applied |
| **Normal-consistency** | mesh coherence 0.81 → 0.92 | Yes (helmet mesh) | ✅ applied |
| **Appearance embeddings** | unlocks multi-clip | High — pro camera + iPhone | ✅ applied (multi-clip) |
| Bilateral-grid appearance | tie (no spatial variation) | Possible (pro/iPhone vignetting) | ⏭️ low priority |
| **Pose optimization** | −1 dB (poses already 0.9 px) | Hypothesis falsified — hurts again | ❌ −0.54 dB, CI [−0.71, −0.36] |
| **2DGS** surface backend | FAILED, 13 dB (single-sided) | **Works** — a real orbit supplies the normals | ✅ parity w/ 3DGS, `dist_lambda=0` |
| Object box (attract to subject) | n/a | tested | ❌ no sharper helmet + smearing (data-limited) |
| Generative diffusion prior | worse at every strength | Same regime, and helmet is data-limited | ⏭️ skip |
| Object-region metrics | n/a | Helmet fills the frame (mask ~100 %) | ➖ limited |
| TSDF mesh + object crop | works (soft) | Yes | ✅ applied |

## The three findings that motivated the transfer tests

Because the Casque geometry is the Pavillon's opposite, three Pavillon results are not
safe to assume and are being measured rather than copied:

1. **Capacity inverts — CONFIRMED.** The Pavillon optimum was *375 k*: surplus Gaussians
   overfit its low-parallax views. An orbit supplies real multi-view constraint, so the
   overfitting pressure is weaker and the optimum should be *higher*. It is:

   | `cap_max` | 750 k | 1.5 M | 3 M |
   |---|---|---|---|
   | test PSNR | 19.49 | **20.35** | 20.76 |
   | SSIM | 0.8585 | 0.8636 | **0.8653** |
   | LPIPS | 0.2892 | 0.2710 | **0.2612** |

   Paired per-view tests over the 13 test views: 750 k → 1.5 M is **+0.86 dB, 95 % CI
   [+0.34, +1.39]** (10/13 views improve) — real. 1.5 M → 3 M is **+0.41 dB, CI
   [−0.45, +1.26]** (8/13) — **a tie**. So the Casque curve *rises then plateaus*
   rather than turning, and **1.5 M is the operating point**: 3 M doubles the model for
   a gain indistinguishable from noise.

   Copying the Pavillon's 375 k here would have cost well over a decibel. The rule that
   transfers is not the number but the method: **sweep capacity per capture, and read
   the paired CI rather than the mean.**

2. **2DGS works here — CONFIRMED, after one loss weight was fixed.** It collapsed on the
   Pavillon because a surface-aligned disk must recover a normal that a single-sided
   capture never observes. A helmet orbit observes every surface from many directions,
   which is exactly the evidence 2DGS needs.

   The first run *looked* like the same failure — 15.79 dB vs 20.35 for 3DGS. The
   trajectory said otherwise: it was at **24.6 dB and still climbing** until the
   distortion and normal regularizers engaged at iteration 7000, then decayed
   monotonically for 23 k more iterations. Setting `dist_lambda: 0` recovers
   **+4.40 dB, CI [+2.32, +6.48]** (12/13 views) and lands at 20.19 — **a statistical
   tie with 3DGS** (−0.16 dB, CI [−1.26, +0.94]).

   ![2DGS distortion collapse](assets/casque/2dgs_distortion.png)

   Parity is the *success* condition for a surface method: the same photometric quality
   while producing a surface-aligned mesh. The distortion loss concentrates ray weight
   onto one surface, which suits a bounded object — but masks are off here, so the model
   must also explain a full room at varied depths, and that term fights it.

   **The methodological point:** the final metric alone supported "2DGS does not work on
   this capture." The trajectory supported "2DGS works and one weight was wrong." Those
   are opposite engineering decisions, and only the trajectory distinguishes them.

3. **Pose refinement — hypothesis FALSIFIED.** It cost 1 dB on the Pavillon, which we
   attributed to GLOMAP's poses already being sub-pixel (0.92 px). The mixed pro+iPhone
   Casque set registers at 1.57 px, so there should have been genuine calibration error
   to recover. There was not: **−0.54 dB, CI [−0.71, −0.36]**, improving only 9 of 53
   views. Pose refinement is now a negative on *both* captures, and the "poses were
   already too good" explanation does not survive — a better hypothesis is that our
   SE(3) refinement trades multi-view consistency for per-view photometric fit.

## Findings that transferred as-is

- **GLOMAP, the trainer fixes, anti-floater, MCMC, normal-consistency and appearance
  embeddings** all apply unchanged and help. The checkerboard even makes GLOMAP's job
  *easier* here (shared features across all clips) than on the Pavillon.
- **Masks are the exception to "object capture ⇒ mask it".** rembg was unreliable on the
  chrome/plume/stand subject, and masking would have discarded the checkerboard — the
  best SfM features. The Casque is reconstructed unmasked and the helmet isolated in 3D.

## The meta-lesson

The same discipline transferred even where the techniques did not: **judge by looking,
report negatives, and measure rather than assume.** The object box, generative priors,
and (on the Pavillon) 2DGS and pose refinement are all documented negatives — each cost
a probe, not a project, precisely because they were tested rather than trusted. The
per-capture answer is the point; the framework is the same.
