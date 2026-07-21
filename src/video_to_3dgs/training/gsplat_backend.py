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
                   near, far, with_depth=False):
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
            render_mode="RGB+ED" if with_depth else "RGB",
        )
        if with_depth:
            # RGB+ED -> (...,:3) colour, (...,3:4) expected (median) depth
            return renders[..., :3], renders[..., 3:4], alphas, info
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

        # Per-image appearance model (multi-clip exposure/WB reconciliation).
        # Kept OUT of `optimizers`, which the densification strategy owns and
        # indexes per-Gaussian — these parameters are per-IMAGE, not per-Gaussian.
        app_model = app_opt = None
        clip_to_train_idx: dict[str, list[int]] = {}
        clip_key = None
        if cfg.appearance_embedding:
            from .appearance import AppearanceModel, clip_key
            app_model = AppearanceModel(len(train_ds), dim=cfg.appearance_dim).to(device)
            app_opt = torch.optim.Adam(app_model.parameters(), lr=cfg.appearance_lr)
            for j, s in enumerate(train_ds.samples):
                clip_to_train_idx.setdefault(clip_key(s.name), []).append(j)
            log.info("appearance embeddings ON: %d images x %d dims across %d clip(s): %s",
                     len(train_ds), cfg.appearance_dim, len(clip_to_train_idx),
                     ", ".join(f"{k}({len(v)})" for k, v in sorted(clip_to_train_idx.items())))

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

        # Room bounding box (AABB of cameras+points, expanded) to keep Gaussians
        # inside the captured room and kill runaway floaters.
        box_lo = box_hi = None
        if cfg.bounds.enabled:
            from .dataset import _center_from_viewmat
            pts = [train_ds.points] if len(train_ds.points) else []
            cams = np.array([_center_from_viewmat(s.viewmat) for s in train_ds.samples])
            allp = np.concatenate(pts + [cams], axis=0) if pts else cams
            lo, hi = allp.min(0), allp.max(0)
            m = cfg.bounds.margin * (hi - lo)
            box_lo = torch.tensor(lo - m, dtype=torch.float32, device=device)
            box_hi = torch.tensor(hi + m, dtype=torch.float32, device=device)
            log.info("room box (normalized): lo=%s hi=%s", np.round(lo - m, 2), np.round(hi + m, 2))

        # appearance state travels in the checkpoint's `extra` (it is not a
        # per-Gaussian tensor, so it cannot ride in params/optimizers)
        def _ckpt_extra(base: dict | None = None) -> dict:
            e = dict(base or {})
            if app_model is not None:
                e["appearance"] = {k: v.detach().cpu()
                                   for k, v in app_model.state_dict().items()}
            return e

        # resume
        start_step = 0
        if ctx.resume:
            latest = find_latest_valid(ckpt_dir)
            if latest is not None:
                start_step = load_checkpoint(latest, params, optimizers) + 1
                log.info("resumed from %s at step %d", latest.name, start_step)
                if app_model is not None:
                    try:
                        st = torch.load(latest, map_location="cpu", weights_only=False)
                        app_sd = (st.get("extra") or {}).get("appearance")
                        if app_sd and app_sd["embed.weight"].shape[0] == app_model.n_images:
                            app_model.load_state_dict(app_sd)
                            log.info("resumed appearance embeddings")
                        elif app_sd:
                            log.warning("appearance embeddings in checkpoint cover %d images "
                                        "but dataset has %d; starting them fresh",
                                        app_sd["embed.weight"].shape[0], app_model.n_images)
                    except Exception as e:  # noqa: BLE001
                        log.warning("could not restore appearance embeddings: %s", e)

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

        # Monocular-depth prior (A3): precompute per-frame depth targets once.
        depth_bank = None
        if cfg.depth_prior.enabled:
            from .depth_priors import DepthPriorBank, pearson_depth_loss
            # cache at the dataset level (run_dir) so every training run on these
            # frames reuses the depth maps instead of recomputing them.
            depth_bank = DepthPriorBank.build(
                train_ds.samples, layout.run_dir / "depth_prior",
                cfg.depth_prior.model, device)
            if depth_bank is None:
                log.warning("depth prior requested but unavailable; training photometric-only")

        max_iters = cfg.max_iterations
        # Standard 3DGS position-LR decay: without it the Gaussian centers keep
        # jittering at the full LR for the whole run and never settle to sharp
        # detail (renders come out as blurry fog). Decay the means LR ~100x over
        # training (gamma so lr(final) = lr(0) * 0.01). Resume-safe via last_epoch.
        means_sched = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max(max_iters, 1)),
            last_epoch=start_step - 1)
        order = np.arange(len(train_ds))
        np.random.shuffle(order)
        cursor = 0
        t0 = time.time()
        status = "COMPLETED"
        best_val = {"psnr": None}
        cap_frozen = False
        step = min(start_step, max_iters) - 1   # defined even if the loop body never runs
        if start_step >= max_iters:
            log.info("resumed checkpoint already at step %d >= max_iters %d; skipping training",
                     start_step, max_iters)

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

            want_depth = depth_bank is not None and step >= cfg.depth_prior.start_iter
            if want_depth:
                rgb, depth_map, alphas, info = self._rasterize(
                    gsplat, params, vm, K, w, h, sh_now, near, far, with_depth=True)
                render = rgb[0]
            else:
                renders, alphas, info = self._rasterize(gsplat, params, vm, K, w, h, sh_now,
                                                        near, far)
                render = renders[0]
            # per-image photometric correction (identity at init; cannot encode
            # geometry, only a global affine colour change)
            if app_model is not None:
                render = app_model(idx, render)
            loss, l1, ssim_val = photometric_loss(render, gt, mask_t, cfg.l1_lambda,
                                                  cfg.ssim_lambda)
            depth_loss_val = 0.0
            if want_depth:
                tgt = depth_bank.get(train_ds.samples[idx].name, device, (h, w))
                if tgt is not None:
                    dl = pearson_depth_loss(depth_map[0, ..., 0], tgt, mask_t)
                    loss = loss + cfg.depth_prior.lambda_depth * dl
                    depth_loss_val = float(dl.detach())
            health.check_loss(float(loss.detach()), step)

            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            loss.backward()

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            if app_opt is not None:
                app_opt.step()
                app_opt.zero_grad(set_to_none=True)
            means_sched.step()   # decay position LR so Gaussians settle (sharpness)
            # Densify/prune — but FREEZE growth once the cap is reached instead of
            # crashing. gsplat's DefaultStrategy has no built-in cap, so gate it here.
            if params["means"].shape[0] < dcfg.cap_max:
                strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                            packed=False)
            elif not cap_frozen:
                log.warning("gaussian cap %d reached at step %d; freezing densification "
                            "and continuing to optimize", dcfg.cap_max, step)
                cap_frozen = True

            # Regularization: suppress out-of-room + oversized 'flying' Gaussians by
            # driving their opacity to ~0 (no size change -> no strategy-state conflict);
            # the DefaultStrategy's opacity prune then removes them during densification.
            if step > 0:
                with torch.no_grad():
                    kill = None
                    if box_lo is not None and step % cfg.bounds.prune_every == 0:
                        m = params["means"]
                        kill = ((m < box_lo) | (m > box_hi)).any(dim=1)
                    if cfg.floater.enabled and step % cfg.floater.prune_every == 0:
                        huge = torch.exp(params["scales"]).max(dim=1).values > \
                            cfg.floater.max_scale_frac * scene_scale
                        kill = huge if kill is None else (kill | huge)
                    if kill is not None and bool(kill.any()):
                        params["opacities"].data[kill] = -20.0  # sigmoid(-20) ~ 2e-9

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
                    "depth_loss": depth_loss_val,
                    "appearance_drift": app_model.drift() if app_model is not None else 0.0,
                })

            # validation
            if val_ds and (step % cfg.validation_interval == 0 and step > 0 or step == max_iters - 1):
                def _val_render(i, _step=step):
                    r = self._rasterize(gsplat, params, *val_ds.camera_tensors(i, device),
                                        min(cfg.sh_degree, _step // 1000), near, far)[0][0]
                    if app_model is None:
                        return r
                    # score under the mean appearance of the view's OWN source clip
                    # (clip identity + that clip's TRAINING images only — the
                    # held-out pixels are never used, so nothing leaks). Averaging
                    # across clips penalises every view once the clips differ.
                    return app_model.canonical_for(
                        r, clip_to_train_idx.get(clip_key(val_ds.samples[i].name)))

                vres = evaluate_split(
                    _val_render,
                    val_ds, device, out_dir=renders_dir / f"val_{step:07d}",
                    compute_lpips=False, masked=cfg.use_masks, max_images=cfg.val_render_count)
                metrics.log(step, {"psnr": vres["psnr"], "ssim": vres["ssim"]}, kind="val")
                log.info("step %d val: psnr=%s ssim=%s n_gauss=%d", step, vres["psnr"],
                         vres["ssim"], n_gauss)
                if best_val["psnr"] is None or (vres["psnr"] or 0) > best_val["psnr"]:
                    best_val = {"psnr": vres["psnr"], "step": step}
                if health.check_improvement(vres["psnr"] or 0.0, step):
                    log.info("early stopping at step %d (no improvement)", step)
                    save_checkpoint(ckpt_dir, params, optimizers, step, _ckpt_extra({"early_stop": True}))
                    break

            # checkpoint
            if step > 0 and step % cfg.checkpoint_interval == 0:
                save_checkpoint(ckpt_dir, params, optimizers, step, _ckpt_extra())

            # preemption
            if preempt():
                log.warning("preemption signal received; checkpointing at step %d", step)
                save_checkpoint(ckpt_dir, params, optimizers, step, _ckpt_extra({"preempted": True}))
                status = "PREEMPTED"
                break

        # Final hard prune (only on clean completion; a preempted run resumes and must
        # keep its optimizer state intact). Removes faint + out-of-room Gaussians for a
        # clean, floater-free deliverable model.
        if status == "COMPLETED" and (cfg.floater.enabled or box_lo is not None):
            with torch.no_grad():
                keep = torch.sigmoid(params["opacities"].squeeze(-1)) >= cfg.floater.min_opacity
                if box_lo is not None:
                    m = params["means"]
                    keep = keep & ~((m < box_lo) | (m > box_hi)).any(dim=1)
                n_before = int(params["means"].shape[0])
                if bool((~keep).any()):
                    idx = keep.nonzero(as_tuple=True)[0]
                    import torch.nn as _nn
                    for k in list(params.keys()):
                        params[k] = _nn.Parameter(params[k].data[idx])
                    log.info("final prune: %d -> %d gaussians (%.1f%% floaters removed)",
                             n_before, len(idx), 100 * (1 - len(idx) / max(n_before, 1)))

        final = save_checkpoint(ckpt_dir, params, optimizers, min(step, max_iters - 1),
                                _ckpt_extra({"final": True}))
        metrics.close()
        n_final = int(params["means"].shape[0])
        log.info("training %s at step %d: %d gaussians, best_val_psnr=%s",
                 status, step, n_final, best_val.get("psnr"))
        return TrainResult(final_checkpoint=final,
                           metrics={"best_val_psnr": best_val.get("psnr"),
                                    "final_step": step, "n_gaussians": n_final},
                           n_gaussians=n_final, status=status)

    # ------------------------------------------------------------------ #
    def export_mesh(self, ctx: TrainContext, checkpoint: Path, out: Path, *,
                    voxel: float | None = None, sdf_trunc_scale: float = 5.0,
                    depth_trunc: float = 3.0, alpha_min: float = 0.5) -> "Path | None":
        """TSDF-fuse the model's rendered depth over the training cameras -> mesh.

        The subject here is a carved relief, so a triangle mesh is often more useful
        than a radiance field. 2DGS is the principled route to surfaces, but it fails
        on this single-sided capture (flat renders, PSNR ~13), so we fuse depth from
        the 3D Gaussian model instead. Caveat worth stating: volumetric primitives
        give a *biased* expected depth — a Gaussian straddling a surface contributes
        mass in front of and behind it — so this mesh is geometrically softer than a
        surface-aligned method would produce.

        Two details matter for usable output:
        * **Alpha masking.** Where little opacity accumulates (background, unobserved
          regions) the composited depth is meaningless, not merely noisy. Those pixels
          are dropped rather than fused, otherwise they drag spurious surfaces into
          the volume.
        * **Scale-aware voxels.** The scene lives in normalized units, so a fixed
          voxel size is meaningless across captures; it is derived from the robust
          point extent unless overridden.
        """
        try:
            import open3d as o3d
        except Exception as e:  # noqa: BLE001
            ctx.logger.warning("open3d not available; skipping mesh export: %s", e)
            return None
        import torch
        import gsplat

        from .gaussians import create_splats

        if not torch.cuda.is_available():
            ctx.logger.warning("mesh export needs CUDA to rasterize depth; skipping")
            return None
        device = "cuda"
        ds = ColmapDataset(ctx.layout, "train", downscale=1)
        params, _ = create_splats(ds.points, ds.point_colors, ctx.train_cfg.sh_degree, device)
        load_checkpoint(checkpoint, params, {})

        near = 0.01
        if ctx.layout.normalize_transform.exists():
            import json
            near = max(1e-3, float(json.loads(ctx.layout.normalize_transform.read_text())
                                   .get("near", 0.01)) * 0.5)

        if voxel is None:
            lo, hi = np.percentile(ds.points, 2, axis=0), np.percentile(ds.points, 98, axis=0)
            voxel = max(float(np.max(hi - lo)) / 512.0, 1e-4)   # ~512 voxels across
        sdf_trunc = sdf_trunc_scale * voxel
        ctx.logger.info("TSDF: voxel=%.5f sdf_trunc=%.5f over %d cameras",
                        voxel, sdf_trunc, len(ds.samples))

        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        n_used = 0
        for s in ds.samples:
            vm = torch.from_numpy(s.viewmat.astype(np.float32)).to(device)
            K = torch.from_numpy(s.K.astype(np.float32)).to(device)
            with torch.no_grad():
                rgb, depth, alphas, _ = self._rasterize(
                    gsplat, params, vm, K, s.width, s.height,
                    ctx.train_cfg.sh_degree, near, 1e10, with_depth=True)
            rgb_np = rgb[0].clamp(0, 1).cpu().numpy()
            depth_np = depth[0, ..., 0].cpu().numpy()
            alpha_np = alphas[0, ..., 0].cpu().numpy()
            # drop pixels the model is not confident about — their depth is not a
            # noisy surface estimate, it is no estimate at all
            depth_np = np.where(alpha_np >= alpha_min, depth_np, 0.0).astype(np.float32)
            if float((depth_np > 0).mean()) < 0.01:
                continue

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image((rgb_np * 255).astype(np.uint8)),
                o3d.geometry.Image(depth_np),
                depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
            intr = o3d.camera.PinholeCameraIntrinsic(
                s.width, s.height, s.K[0, 0], s.K[1, 1], s.K[0, 2], s.K[1, 2])
            vol.integrate(rgbd, intr, s.viewmat.astype(np.float64))
            n_used += 1

        if n_used == 0:
            ctx.logger.warning("no camera produced usable depth; skipping mesh")
            return None

        mesh = vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        # keep only the largest connected component: TSDF of a partially observed
        # scene leaves small floating shells behind
        try:
            idx, counts, _ = mesh.cluster_connected_triangles()
            idx = np.asarray(idx); counts = np.asarray(counts)
            if len(counts):
                mesh.remove_triangles_by_mask(idx != int(counts.argmax()))
                mesh.remove_unreferenced_vertices()
        except Exception as e:  # noqa: BLE001
            ctx.logger.warning("mesh component filtering skipped: %s", e)

        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out), mesh)
        ctx.logger.info("mesh: %d verts, %d tris from %d/%d cameras -> %s",
                        len(mesh.vertices), len(mesh.triangles), n_used, len(ds.samples),
                        out.name)
        return out

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
