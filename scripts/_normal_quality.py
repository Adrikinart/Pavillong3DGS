"""How well-defined are the per-Gaussian normals the specular head reflects about?

The head reflects the view direction about each Gaussian's axis of LEAST variance. For a
near-isotropic Gaussian that axis is essentially arbitrary, so the reflection direction is
noise and no coherent reflection field can be learned.
"""
import torch, numpy as np
p = "experiments/runs/casque_orbit_07ccd886/trainings/casque_helmet_masked/checkpoints/ckpt_latest.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
sc = torch.exp(d["params"]["scales"])                    # (N,3)
s, _ = torch.sort(sc, dim=-1)                            # ascending
flatness = (s[:, 0] / s[:, 2]).numpy()                   # 0 = disc/needle, 1 = sphere
mid_ratio = (s[:, 0] / s[:, 1]).numpy()                  # how distinct the SMALL axis is
print(f"n = {len(flatness):,}")
for q in (10, 25, 50, 75, 90):
    print(f"  p{q:<2}  smallest/largest = {np.percentile(flatness, q):.3f}   "
          f"smallest/middle = {np.percentile(mid_ratio, q):.3f}")
amb = (mid_ratio > 0.8).mean()
print(f"\nGaussians whose smallest axis is within 20% of the middle axis: {amb:.1%}")
print("For those the 'normal' is close to arbitrary -- any reflection direction computed "
      "from it is noise.")
