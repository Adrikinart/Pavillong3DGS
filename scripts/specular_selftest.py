"""Validate the specular head: no-op at init, and survives densification."""
import numpy as np, torch, gsplat
from video_to_3dgs.training.gsplat_backend import GsplatBackend
from video_to_3dgs.training.gaussians import create_splats
from video_to_3dgs.training.specular import init_specular_bank

dev = "cuda"
torch.manual_seed(0)
n = 5000
pts = np.random.RandomState(0).normal(size=(n, 3)).astype(np.float32) * 0.3
cols = np.random.RandomState(1).rand(n, 3).astype(np.float32)
params, opts = create_splats(pts, cols, 3, dev)

vm = torch.eye(4, device=dev); vm[2, 3] = 3.0
K = torch.tensor([[300., 0, 160.], [0, 300., 120.], [0, 0, 1.]], device=dev)
be = GsplatBackend()

a = be._rasterize(gsplat, params, vm, K, 320, 240, 3, 0.01, 1e10)[0]
import torch.nn as nn
params["sh_spec"] = nn.Parameter(init_specular_bank(n, 2, dev))
b = be._rasterize(gsplat, params, vm, K, 320, 240, 3, 0.01, 1e10, spec_degree=2)[0]

d = (a - b).abs().max().item()
print(f"max |SH path - specular path| at zero init = {d:.3e}")
print("NO-OP AT INIT:", "PASS" if d < 1e-5 else "FAIL")

# non-zero coefficients must actually change the image (i.e. it is wired, not ignored)
with torch.no_grad():
    params["sh_spec"].normal_(0, 0.2)
c = be._rasterize(gsplat, params, vm, K, 320, 240, 3, 0.01, 1e10, spec_degree=2)[0]
print(f"max change with non-zero coeffs = {(a - c).abs().max().item():.3e}",
      "-> ACTIVE" if (a - c).abs().max().item() > 1e-3 else "-> INERT (bug)")

# gradients must reach the bank
params["sh_spec"].grad = None
out = be._rasterize(gsplat, params, vm, K, 320, 240, 3, 0.01, 1e10, spec_degree=2)[0]
out.square().mean().backward()
g = params["sh_spec"].grad
print("GRADIENT:", "PASS" if g is not None and torch.isfinite(g).all() and g.abs().max() > 0
      else "FAIL")

# densification must reindex the extra per-Gaussian tensor along with the rest
opts["sh_spec"] = torch.optim.Adam([params["sh_spec"]], lr=2.5e-3, eps=1e-15)
strat = gsplat.MCMCStrategy(cap_max=n * 2, refine_start_iter=0, refine_every=10,
                            min_opacity=0.005)
st = strat.initialize_state()
for step in range(31):
    r, alph, info = be._rasterize(gsplat, params, vm, K, 320, 240, 3, 0.01, 1e10,
                                  spec_degree=2)
    loss = r.square().mean()
    loss.backward()
    for o in opts.values():
        o.step(); o.zero_grad(set_to_none=True)
    strat.step_post_backward(params, opts, st, step, info, lr=1e-4)
ok = params["sh_spec"].shape[0] == params["means"].shape[0]
print(f"after densification: means={params['means'].shape[0]} "
      f"sh_spec={params['sh_spec'].shape[0]}")
print("DENSIFICATION REINDEX:", "PASS" if ok else "FAIL")
