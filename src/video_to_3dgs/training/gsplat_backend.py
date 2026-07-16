"""Self-contained gsplat training backend.

Built on the *packaged* gsplat APIs (``gsplat.rasterization`` +
``gsplat.DefaultStrategy``) — not on ``examples/simple_trainer.py`` — so it does
not depend on unstable example code. Owns the loop to control checkpointing,
health checks, preemption handling, and structured metrics.
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


class GsplatBackend(TrainingBackend):
    name = "gsplat"

    # ------------------------------------------------------------------ #
    def validate_env(self) -> None:
        try:
            import torch
        except Exception as e:
            raise EnvironmentError_(f"torch not importable: {e}") from e
        if not torch.cuda.is_available():
            raise EnvironmentError_("CUDA not available for gsplat training")
        try:
            import gsplat  # noqa: F401
        except Exception as e:
            raise EnvironmentError_(f"gsplat not importable: {e}") from e
        # arch compatibility: device sm must be in torch's compiled arch list
        p = torch.cuda.get_device_properties(0)
        want = f"{p.major}{p.minor}"
        archs = torch.cuda.get_arch_list()
        if not any(want in a for a in archs):
            raise EnvironmentError_(
                f"device sm_{want} not in torch arch list {archs}; "
                f"install a cu128 build with Blackwell (sm_120) kernels")

    # ------------------------------------------------------------------ #
    def _rasterize(self, gsplat, params, sample_vm, K, width, height, sh_degree_now,
                   near, far):
        import torch
        means = params["means"]
        quats = params["quats"]
        scales = torch.exp(params["scales"])
        opac = torch.sigmoid(params["opacities"])
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N,K,3)
        renders, alphas, info = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
            viewmats=sample_vm[None], Ks=K[None], width=width, height=height,
            sh_degree=sh_degree_now, near_plane=near, far_plane=far,
            packed=False, rasterize_mode="antialiased",
        )
        return renders, alphas, info

    # ------------------------------------------------------------------ #
    def train(self, ctx: TrainContext) -> TrainResult:  # noqa: C901
        import torch
        import gsplat

        log = ctx.logger
        cfg = ctx.train_cfg
        device = ctx.device
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        layout = ctx.layout
        tr_id = ctx.train_run_id
        ckpt_dir = layout.checkpoints_dir(tr_id)
        renders_dir = layout.renders_dir(tr_id)

        # datasets
        train_ds = ColmapDataset(layout, "train", use_masks=cfg.use_masks,
                                 downscale=cfg.image_downscale)
        val_ds = ColmapDataset(layout, "val", use_masks=cfg.use_masks,
                               downscale=cfg.image_downscale)
        log.info("train views=%d val views=%d points=%d", len(train_ds), len(val_ds),
                 len(train_ds.points))
        scene_scale = train_ds.scene_extent()

        # normalization near/far
        near, far = 0.01, 1e10
        if layout.normalize_transform.exists():
            import json
            d = json.loads(layout.normalize_transform.read_text())
            near = max(1e-3, float(d.get("near", 0.01)) * 0.5)

        # splats + optimizers
        params, optimizers = create_splats(
            train_ds.points, train_ds.point_colors, cfg.sh_degree, device,
            lr_means=cfg.lr_means, scene_scale=max(scene_scale, 1e-3))

        # densification strategy
        dcfg = cfg.densification
        strategy = gsplat.DefaultStrategy(
            prune_opa=dcfg.prune_opacity, grow_grad2d=dcfg.grad_threshold,
            refine_start_iter=dcfg.start_iteration, refine_stop_iter=dcfg.stop_iteration,
            reset_every=dcfg.opacity_reset_interval, refine_every=dcfg.interval,
            verbose=False,
        )
        strategy_state = strategy.initialize_state(scene_scale=max(scene_scale, 1e-3))
        try:
            strategy.check_sanity(params, optimizers)
        except Exception as e:
            log.warning("strategy sanity check: %s", e)

        # resume
        start_step = 0
        if ctx.resume:
            latest = find_latest_valid(ckpt_dir)
            if latest is not None:
                start_step = load_checkpoint(latest, params, optimizers) + 1
                log.info("resumed from %s at step %d", latest.name, start_step)

        metrics = MetricsLogger(layout.metrics_jsonl(tr_id), layout.tensorboard_dir(tr_id),
                                enable_tb=ctx.config.monitoring.tensorboard)
        # Soft cap is enforced by freezing densification (below); the health check
        # is only a genuine-runaway safety net (e.g. NaN-driven), well above the cap.
        health = HealthMonitor(gaussian_cap=int(dcfg.cap_max * 2.5),
                               no_improve_patience=cfg.early_stop_patience)
        preempt = PreemptionHandler()
        preempt.install()

        def render_fn(i: int):
            vm, K, w, h = val_ds.camera_tensors(i, device)
            deg = min(cfg.sh_degree, max(0, start_step // 1000))
            r, _, _ = self._rasterize(gsplat, params, vm, K, w, h, cfg.sh_degree, near, far)
            return r[0]

        max_iters = cfg.max_iterations
        order = np.arange(len(train_ds))
        np.random.shuffle(order)
        cursor = 0
        t0 = time.time()
        status = "COMPLETED"
        best_val = {"psnr": None}
        cap_frozen = False

        for step in range(start_step, max_iters):
            if cursor >= len(order):
                np.random.shuffle(order)
                cursor = 0
            idx = int(order[cursor]); cursor += 1

            gt, mask = train_ds.load_image(idx)
            gt = gt.to(device)
            mask_t = mask.to(device) if (cfg.use_masks and mask is not None) else None
            vm, K, w, h = train_ds.camera_tensors(idx, device)
            sh_now = min(cfg.sh_degree, step // 1000)

            renders, alphas, info = self._rasterize(gsplat, params, vm, K, w, h, sh_now,
                                                    near, far)
            render = renders[0]
            loss, l1, ssim_val = photometric_loss(render, gt, mask_t, cfg.l1_lambda,
                                                  cfg.ssim_lambda)
            health.check_loss(float(loss.detach()), step)

            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            loss.backward()

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            # Densify/prune — but FREEZE growth once the cap is reached instead of
            # crashing. gsplat's DefaultStrategy has no built-in cap, so gate it here.
            if params["means"].shape[0] < dcfg.cap_max:
                strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                            packed=False)
            elif not cap_frozen:
                log.warning("gaussian cap %d reached at step %d; freezing densification "
                            "and continuing to optimize", dcfg.cap_max, step)
                cap_frozen = True

            n_gauss = params["means"].shape[0]
            if step % 50 == 0:
                health.check_gaussian_count(n_gauss, step)
                with torch.no_grad():
                    max_scale = float(torch.exp(params["scales"]).max())
                    mean_opa = float(torch.sigmoid(params["opacities"]).mean())
                health.check_scales(max_scale, step)
                ips = (step - start_step + 1) / max(time.time() - t0, 1e-6)
                metrics.log(step, {
                    "loss": float(loss.detach()), "l1": float(l1), "ssim": float(ssim_val),
                    "psnr": psnr(render.detach(), gt, mask_t), "n_gaussians": n_gauss,
                    "max_scale": max_scale, "mean_opacity": mean_opa,
                    "sh_degree": sh_now, "iters_per_s": round(ips, 2),
                })

            # validation
            if val_ds and (step % cfg.validation_interval == 0 and step > 0 or step == max_iters - 1):
                vres = evaluate_split(
                    lambda i: self._rasterize(gsplat, params, *val_ds.camera_tensors(i, device),
                                              min(cfg.sh_degree, step // 1000), near, far)[0][0],
                    val_ds, device, out_dir=renders_dir / f"val_{step:07d}",
                    compute_lpips=False, masked=cfg.use_masks, max_images=cfg.val_render_count)
                metrics.log(step, {"psnr": vres["psnr"], "ssim": vres["ssim"]}, kind="val")
                log.info("step %d val: psnr=%s ssim=%s n_gauss=%d", step, vres["psnr"],
                         vres["ssim"], n_gauss)
                if best_val["psnr"] is None or (vres["psnr"] or 0) > best_val["psnr"]:
                    best_val = {"psnr": vres["psnr"], "step": step}
                if health.check_improvement(vres["psnr"] or 0.0, step):
                    log.info("early stopping at step %d (no improvement)", step)
                    save_checkpoint(ckpt_dir, params, optimizers, step, {"early_stop": True})
                    break

            # checkpoint
            if step > 0 and step % cfg.checkpoint_interval == 0:
                save_checkpoint(ckpt_dir, params, optimizers, step)

            # preemption
            if preempt():
                log.warning("preemption signal received; checkpointing at step %d", step)
                save_checkpoint(ckpt_dir, params, optimizers, step, {"preempted": True})
                status = "PREEMPTED"
                break

        final = save_checkpoint(ckpt_dir, params, optimizers, min(step, max_iters - 1),
                                {"final": True})
        metrics.close()
        n_final = int(params["means"].shape[0])
        log.info("training %s at step %d: %d gaussians, best_val_psnr=%s",
                 status, step, n_final, best_val.get("psnr"))
        return TrainResult(final_checkpoint=final,
                           metrics={"best_val_psnr": best_val.get("psnr"),
                                    "final_step": step, "n_gaussians": n_final},
                           n_gaussians=n_final, status=status)

    # ------------------------------------------------------------------ #
    def export_ply(self, ctx: TrainContext, checkpoint: Path, out: Path) -> Path:
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        p = state["params"]
        import numpy as _np
        means = p["means"].numpy()
        scales = p["scales"].numpy()
        quats = p["quats"].numpy()
        opac = p["opacities"].numpy().reshape(-1, 1)
        # INRIA/standard-viewer .ply expects channel-major SH: f_dc_{0..2} then
        # f_rest ordered [ch0 coeffs..., ch1..., ch2...]. sh0 is (N,1,3); shN (N,K-1,3).
        sh0 = p["sh0"].numpy().reshape(len(means), -1)                 # (N,3) DC per channel
        shN = _np.transpose(p["shN"].numpy(), (0, 2, 1)).reshape(len(means), -1)  # channel-major
        _write_gaussian_ply(out, means, scales, quats, opac, sh0, shN)
        return out


def _write_gaussian_ply(path: Path, means, scales, quats, opac, sh0, shN) -> None:
    """Write the standard 3DGS .ply (INRIA/gsplat-compatible property layout)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(means)
    normals = np.zeros((n, 3), dtype=np.float32)
    props = ["x", "y", "z", "nx", "ny", "nz"]
    props += [f"f_dc_{i}" for i in range(sh0.shape[1])]
    props += [f"f_rest_{i}" for i in range(shN.shape[1])]
    props += ["opacity"]
    props += [f"scale_{i}" for i in range(scales.shape[1])]
    props += [f"rot_{i}" for i in range(quats.shape[1])]
    data = np.concatenate([means, normals, sh0, shN, opac, scales, quats], axis=1).astype(np.float32)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {p}" for p in props]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())
