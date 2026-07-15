# Object / scene capture guide

Good captures make or break the reconstruction. Follow these when filming.

## Golden rules
- **Keep the object static. Move the camera.** (Turntable = object moves → masks
  mandatory, use `object_turntable.yaml`.)
- **Multiple elevation orbits**: a low ring, an eye-level ring, a high ring.
  Capture the top and the lower sides.
- **Maintain overlap** between consecutive frames (~70–80%). Move smoothly and
  slowly.
- **Avoid pure rotation** (spinning in place) — SfM needs translation (parallax).
- **Avoid motion blur**: move slowly, ensure enough light, high shutter speed.
- **Lock focus, exposure, and white balance** if your camera allows (AE/AF lock).
- **No digital zoom** (changes intrinsics). Use a fixed focal length.
- **Soft, stable, diffuse lighting.** Avoid moving shadows and flicker.
- **Minimize reflections/specularity** and transparent surfaces where possible.
- **Textured background helps SfM** for scenes; for a single object, use masks so
  the background is ignored.

## Practical
- 1–3 minutes per orbit is plenty at 30 fps; the pipeline subsamples frames.
- Prefer even, consistent motion over speed. Redundant near-identical frames are
  filtered out automatically.
- Review sharpness before reconstruction: inspect `frames_filtered/frame_scores.csv`
  and the contact sheets after `filter-frames`.
- iPhone portrait clips carry rotation metadata; the pipeline auto-rotates.

## What the pipeline does with your capture
- Extracts frames (rotation-aware), scores blur/exposure/duplicates, filters
  (with a minimum-coverage safeguard so trajectory coverage is preserved).
- Optionally masks the object (rembg by default).
- Runs COLMAP (sequential matching for video, CPU SIFT by default).
- Validates the reconstruction against quality gates and produces diagnostics.
- Normalizes the scene, makes pose-aware splits, trains, evaluates on held-out
  views, and exports.

## Turntable specifics
- The background is static while the object rotates → COLMAP would lock onto the
  background. Masks are mandatory so only object features are used, and matching
  falls back to exhaustive (no temporal ordering assumption).
