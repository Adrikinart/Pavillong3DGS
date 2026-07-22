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
| **Capacity** sweep | 375 k optimal (*less is more*) | **May INVERT** — an orbit is not sparse-view | 🔬 sweeping 750k/1.5M/3M |
| **MCMC** densification | parity, cleaner control | Yes | ✅ applied |
| **Normal-consistency** | mesh coherence 0.81 → 0.92 | Yes (helmet mesh) | ✅ applied |
| **Appearance embeddings** | unlocks multi-clip | High — pro camera + iPhone | ✅ applied (multi-clip) |
| Bilateral-grid appearance | tie (no spatial variation) | Possible (pro/iPhone vignetting) | ⏭️ low priority |
| **Pose optimization** | −1 dB (poses already 0.9 px) | **May help** — mixed-camera poses are 1.57 px | 🔬 to test |
| **2DGS** surface backend | FAILED, 13 dB (single-sided) | **Should work** — a real orbit | 🔬 under test |
| Object box (attract to subject) | n/a | tested | ❌ no sharper helmet + smearing (data-limited) |
| Generative diffusion prior | worse at every strength | Same regime, and helmet is data-limited | ⏭️ skip |
| Object-region metrics | n/a | Helmet fills the frame (mask ~100 %) | ➖ limited |
| TSDF mesh + object crop | works (soft) | Yes | ✅ applied |

## The three findings that motivated the transfer tests

Because the Casque geometry is the Pavillon's opposite, three Pavillon results are not
safe to assume and are being measured rather than copied:

1. **Capacity may invert.** The Pavillon optimum was *375 k* — surplus Gaussians
   overfit its low-parallax views. An orbit supplies real multi-view constraint, so the
   overfitting pressure is weaker and the optimum should be *higher*. This is the single
   most important transfer test: if the Pavillon "less is more" held blindly it would
   under-fit the helmet.

2. **2DGS should work.** It collapsed on the Pavillon because a surface-aligned disk
   must recover a normal that a single-sided capture never observes. A helmet orbit
   observes every surface from many directions — exactly the evidence 2DGS needs — so it
   is the natural route to the helmet mesh here.

3. **Pose refinement may help.** It cost 1 dB on the Pavillon *because* GLOMAP's poses
   were already sub-pixel (0.92 px). The mixed pro+iPhone Casque set registers at
   1.57 px, so there may be genuine calibration error to recover this time.

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
