"""Does a generative diffusion prior actually help THIS reconstruction? (diagnostic)

Fail-fast gate before investing in any diffusion-based method (FixingGS, Difix3D,
SparseGS score distillation). It refines the held-out RENDERS of a trained model with
an off-the-shelf image diffusion model (SD-Turbo, img2img, low strength) and measures
whether that moves them CLOSER to ground truth (real information added) or FURTHER
(hallucination). If post-hoc refinement of already-good renders worsens metrics, a
training-time distillation of the same prior would only bake the hallucination in.

Result on the Pavillon panel (226 views, base PSNR 24.92): refinement is monotonically
worse on PSNR/SSIM/LPIPS at every strength -> generative priors do NOT help here. These
methods target the extreme-sparse regime (~3 views) where the reconstruction is broken;
a good reconstruction has no gap for the prior to fill, only detail for it to corrupt.

Env notes: needs `diffusers`; on this cluster prepend the conda env's libstdc++
(LD_LIBRARY_PATH=$ENV/lib) and load the model from its snapshot path with
local_files_only=True (AutoPipeline does a Hub call even for a cached model otherwise).

Usage (GPU node): python scripts/diffusion_refine_test.py <dataset_id> <train_run_id>
"""
import glob, numpy as np, torch
from pathlib import Path
from PIL import Image
from diffusers import AutoPipelineForImage2Image
import sys; sys.path.insert(0, "src")
from video_to_3dgs.core.paths import RunLayout
from video_to_3dgs.training.dataset import ColmapDataset
from video_to_3dgs.training.losses import ssim as SSIM
import lpips as _lp

snap = sorted(glob.glob(str(Path.home()/".cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/*/")))[0]
pipe = AutoPipelineForImage2Image.from_pretrained(snap, torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True).to("cuda"); pipe.set_progress_bar_config(disable=True)
lpnet = _lp.LPIPS(net="alex").cuda().eval()
def met(r, g):
    r=torch.tensor(r,device="cuda"); g=torch.tensor(g,device="cuda")
    ps=10*torch.log10(1/((r-g)**2).mean().clamp_min(1e-12))
    ss=SSIM(r.permute(2,0,1), g.permute(2,0,1))
    lp=lpnet((r.permute(2,0,1)[None]*2-1).half(),(g.permute(2,0,1)[None]*2-1).half()).item()
    return float(ps), float(ss), float(lp)

lay = RunLayout(runs_root=Path("experiments/runs"), dataset_id=dsid)
ds = ColmapDataset(lay, "test", use_masks=False)
by = {Path(s.name).stem: i for i,s in enumerate(ds.samples)}
rdir = lay.renders_dir(sys.argv[2] if len(sys.argv)>2 else "gsplat_hidetail_cap375k")/"eval_test"
strengths=[0.10,0.20,0.30]
acc={("raw",):[]}; [acc.setdefault((s,),[]) for s in strengths]
for f in sorted(rdir.glob("*.png")):
    if f.stem not in by: continue
    g,_ = ds.load_image(by[f.stem]); g=g.cpu().numpy()
    H,W=g.shape[:2]
    r0=np.asarray(Image.open(f).convert("RGB").resize((W,H)),np.float32)/255
    acc[("raw",)].append(met(r0,g))
    img=Image.open(f).convert("RGB")
    for s in strengths:
        out=pipe(prompt="sharp photo of a carved wooden panel, fine engraved detail",
                 image=img, strength=s, num_inference_steps=20, guidance_scale=0.0).images[0]
        rr=np.asarray(out.resize((W,H)),np.float32)/255
        acc[(s,)].append(met(rr,g))
for k,v in acc.items():
    v=np.array(v); tag="raw render" if k[0]=="raw" else f"refine s={k[0]}"
    print(f"  {tag:14s}: PSNR {v[:,0].mean():5.2f}  SSIM {v[:,1].mean():.4f}  LPIPS {v[:,2].mean():.4f}")
print("VS-GT comparison done (n=%d test views)"%len(acc[("raw",)]))
