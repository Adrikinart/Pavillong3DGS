"""2D Gaussian Splatting backend (surface-aligned) via gsplat's native 2DGS.

2DGS represents the scene with oriented flat disks that hug surfaces, giving much
better geometry + normals than 3D Gaussians and enabling clean mesh extraction —
ideal for a bas-relief / carved-surface object. Adds the 2DGS regularizers
(normal consistency + depth distortion) on top of the photometric loss, and uses
the ``gradient_2dgs`` densification key. Mesh extraction (TSDF) lives in
``export_mesh`` (needs Open3D).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import EnvironmentError_
from .backend import TrainContext, TrainingBackend, TrainResult
from .checkpoint import find_latest_valid, load_checkpoint, save_checkpoint
from .dataset import ColmapDataset
from .gaussians import create_splats
from .health import HealthMonitor
from .losses import photometric_loss, psnr
from .metrics import MetricsLogger
from .signals import PreemptionHandler
from .validation import evaluate_split


class Gsplat2DGSBackend(TrainingBackend):
    name = "2dgs"

    def validate_env(self) -> None:
        try:
            import torch
            import gsplat
        except Exception as e:
            raise EnvironmentError_(f"torch/gsplat not importable: {e}") from e
        if not torch.cuda.is_available():
            raise EnvironmentError_("CUDA not available for 2DGS training")
        if not hasattr(gsplat, "rasterization_2dgs"):
            raise EnvironmentError_("installed gsplat lacks rasterization_2dgs; upgrade gsplat")

    # ------------------------------------------------------------------ #
    def _rasterize(self, gsplat, params, vm, K, w, h, sh_deg, near, far):
        import torch
        means = params["means"]
        quats = params["quats"]
        scales = torch.exp(params["scales"])
        opac = torch.sigmoid(params["opacities"])
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        out = gsplat.rasterization_2dgs(
            means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
            viewmats=vm[None], Ks=K[None], width=w, height=h, sh_degree=sh_deg,
            near_plane=near, far_plane=far, render_mode="RGB+ED", packed=False)
        # (colors, alphas, normals, surf_normals, distort, median, meta)
        return out

    # ------------------------------------------------------------------ #
    def train(self, ctx: TrainContext) -> TrainResult:  # noqa: C901
        import torch
        import gsplat

        log = ctx.logger
        cfg = ctx.train_cfg
        device = ctx.device
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        layout, tr = ctx.layout, ctx.train_run_id
        ckpt_dir = layout.checkpoints_dir(tr)

        train_ds = ColmapDataset(layout, "train", use_masks=cfg.use_masks,
                                 downscale=cfg.image_downscale)
        val_ds = ColmapDataset(layout, "val", use_masks=cfg.use_masks,
                               downscale=cfg.image_downscale)
        scene_scale = train_ds.scene_extent()
        log.info("2DGS train views=%d val=%d points=%d scene_scale=%.3f",
                 len(train_ds), len(val_ds), len(train_ds.points), scene_scale)

        near = 0.01
        if layout.normalize_transform.exists():
            import json
            near = max(1e-3, float(json.loads(layout.normalize_transform.read_text())
                                   .get("near", 0.01)) * 0.5)

        params, optimizers = create_splats(
            train_ds.points, train_ds.point_colors, cfg.sh_degree, device,
            lr_means=cfg.lr_means, scene_scale=max(scene_scale, 1e-3))

        dcfg = cfg.densification
        strategy = gsplat.DefaultStrategy(
            prune_opa=dcfg.prune_opacity, grow_grad2d=dcfg.grad_threshold,
            refine_start_iter=dcfg.start_iteration, refine_stop_iter=dcfg.stop_iteration,
            reset_every=dcfg.opacity_reset_interval, refine_every=dcfg.interval,
            key_for_gradient="gradient_2dgs", verbose=False)
        state = strategy.initialize_state(scene_scale=max(scene_scale, 1e-3))

        start_step = 0
        if ctx.resume:
            latest = find_latest_valid(ckpt_dir)
            if latest is not None:
                start_step = load_checkpoint(latest, params, optimizers) + 1
                log.info("resumed 2DGS from %s at step %d", latest.name, start_step)

        means_sched = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max(cfg.max_iterations, 1)),
            last_epoch=start_step - 1)
        metrics = MetricsLogger(layout.metrics_jsonl(tr), layout.tensorboard_dir(tr),
                                enable_tb=ctx.config.monitoring.tensorboard)
        health = HealthMonitor(gaussian_cap=int(dcfg.cap_max * 2.5),
                               no_improve_patience=cfg.early_stop_patience)
        preempt = PreemptionHandler(); preempt.install()

        # 2DGS regularizer weights (ramped in), per the 2DGS paper conventions
        opts = cfg.backend_opts or {}
        w_normal = float(opts.get("normal_lambda", 0.05))
        w_dist = float(opts.get("dist_lambda", 100.0))
        normal_start = int(opts.get("normal_start_iter", 7000))
        dist_start = int(opts.get("dist_start_iter", 3000))

        max_iters = cfg.max_iterations
        order = np.arange(len(train_ds)); np.random.shuffle(order); cursor = 0
        t0 = time.time(); status = "COMPLETED"; best_val = {"psnr": None}
        cap_frozen = False
        step = min(start_step, max_iters) - 1

        for step in range(start_step, max_iters):
            if cursor >= len(order):
                np.random.shuffle(order); cursor = 0
            idx = int(order[cursor]); cursor += 1
            gt, mask = train_ds.load_image(idx); gt = gt.to(device)
            mask_t = mask.to(device) if (cfg.use_masks and mask is not None) else None
            vm, K, w, h = train_ds.camera_tensors(idx, device)
            sh_now = min(cfg.sh_degree, step // 1000)

            colors, alphas, normals, surf_normals, distort, median, info = \
                self._rasterize(gsplat, params, vm, K, w, h, sh_now, near, 1e10)
            render = colors[0, ..., :3]
            loss, l1, ssim_val = photometric_loss(render, gt, mask_t, cfg.l1_lambda,
                                                  cfg.ssim_lambda)
            # normal consistency (align splats to surface) + depth distortion
            reg = {}
            if step >= dist_start:
                dist_loss = distort.mean()
                loss = loss + w_dist * dist_loss
                reg["dist"] = float(dist_loss.detach())
            if step >= normal_start:
                normal_loss = (1.0 - (normals * surf_normals).sum(dim=-1)).mean()
                loss = loss + w_normal * normal_loss
                reg["normal"] = float(normal_loss.detach())
            health.check_loss(float(loss.detach()), step)

            strategy.step_pre_backward(params, optimizers, state, step, info)
            loss.backward()
            for opt in optimizers.values():
                opt.step(); opt.zero_grad(set_to_none=True)
            means_sched.step()
            if params["means"].shape[0] < dcfg.cap_max:
                strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
            elif not cap_frozen:
                log.warning("2DGS gaussian cap %d reached at step %d; freezing densification",
                            dcfg.cap_max, step); cap_frozen = True

            n_gauss = params["means"].shape[0]
            if step % 50 == 0:
                health.check_gaussian_count(n_gauss, step)
                ips = (step - start_step + 1) / max(time.time() - t0, 1e-6)
                metrics.log(step, {"loss": float(loss.detach()), "l1": float(l1),
                                   "ssim": float(ssim_val), "psnr": psnr(render.detach(), gt, mask_t),
                                   "n_gaussians": n_gauss, "sh_degree": sh_now,
                                   "iters_per_s": round(ips, 2), **reg})

            if val_ds and ((step % cfg.validation_interval == 0 and step > 0) or step == max_iters - 1):
                vres = evaluate_split(
                    lambda i: self._rasterize(gsplat, params, *val_ds.camera_tensors(i, device),
                                              min(cfg.sh_degree, step // 1000), near, 1e10)[0][0, ..., :3],
                    val_ds, device, out_dir=layout.renders_dir(tr) / f"val_{step:07d}",
                    compute_lpips=False, masked=cfg.use_masks, max_images=cfg.val_render_count)
                metrics.log(step, {"psnr": vres["psnr"], "ssim": vres["ssim"]}, kind="val")
                log.info("2DGS step %d val: psnr=%s ssim=%s n_gauss=%d", step, vres["psnr"],
                         vres["ssim"], n_gauss)
                if best_val["psnr"] is None or (vres["psnr"] or 0) > best_val["psnr"]:
                    best_val = {"psnr": vres["psnr"], "step": step}

            if step > 0 and step % cfg.checkpoint_interval == 0:
                save_checkpoint(ckpt_dir, params, optimizers, step)
            if preempt():
                save_checkpoint(ckpt_dir, params, optimizers, step, {"preempted": True})
                status = "PREEMPTED"; break

        final = save_checkpoint(ckpt_dir, params, optimizers, min(step, max_iters - 1),
                                {"final": True})
        metrics.close()
        n_final = int(params["means"].shape[0])
        log.info("2DGS training %s at step %d: %d gaussians, best_val_psnr=%s",
                 status, step, n_final, best_val.get("psnr"))
        return TrainResult(final_checkpoint=final,
                           metrics={"best_val_psnr": best_val.get("psnr"),
                                    "final_step": step, "n_gaussians": n_final},
                           n_gaussians=n_final, status=status)

    # ------------------------------------------------------------------ #
    def export_ply(self, ctx: TrainContext, checkpoint: Path, out: Path) -> Path:
        from .gsplat_backend import GsplatBackend
        return GsplatBackend().export_ply(ctx, checkpoint, out)

    def export_mesh(self, ctx: TrainContext, checkpoint: Path, out: Path,
                    voxel: float = 0.004, sdf_trunc: float = 0.02,
                    depth_trunc: float = 3.0) -> Path | None:
        """TSDF-fuse rendered depth over the training cameras -> triangle mesh."""
        try:
            import open3d as o3d
        except Exception as e:
            ctx.logger.warning("open3d not available; skipping mesh export: %s", e)
            return None
        import torch
        import gsplat

        from .gaussians import create_splats

        device = ctx.device if torch.cuda.is_available() else "cpu"
        ds = ColmapDataset(ctx.layout, "train", downscale=1)
        params, _ = create_splats(ds.points, ds.point_colors, ctx.train_cfg.sh_degree, device)
        load_checkpoint(checkpoint, params, {})
        near = 0.01
        if ctx.layout.normalize_transform.exists():
            import json
            near = max(1e-3, float(json.loads(ctx.layout.normalize_transform.read_text())
                                   .get("near", 0.01)) * 0.5)

        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        n = 0
        for s in ds.samples:
            vm = torch.from_numpy(s.viewmat.astype(np.float32)).to(device)
            K = torch.from_numpy(s.K.astype(np.float32)).to(device)
            with torch.no_grad():
                out = self._rasterize(gsplat, params, vm, K, s.width, s.height,
                                      ctx.train_cfg.sh_degree, near, 1e10)
            rgb = out[0][0, ..., :3].clamp(0, 1).cpu().numpy()
            depth = out[0][0, ..., 3].cpu().numpy()
            color_im = o3d.geometry.Image((rgb * 255).astype(np.uint8))
            depth_im = o3d.geometry.Image(depth.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_im, depth_im, depth_scale=1.0, depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False)
            intr = o3d.camera.PinholeCameraIntrinsic(
                s.width, s.height, s.K[0, 0], s.K[1, 1], s.K[0, 2], s.K[1, 2])
            vol.integrate(rgbd, intr, s.viewmat.astype(np.float64))
            n += 1
        mesh = vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out), mesh)
        ctx.logger.info("2DGS mesh: %d verts, %d tris from %d views -> %s",
                        len(mesh.vertices), len(mesh.triangles), n, out)
        return out
