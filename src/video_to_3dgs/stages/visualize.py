"""Stage: generate figures + videos + aggregated metrics from a trained run.

Decoupled from training: operates on the checkpoint, metrics.jsonl and eval
renders, so it can be re-run standalone (`visualize`) or as the final step of
`run-all`. Each figure/video is best-effort — a failure is logged and skipped,
never fails the whole stage.
"""

from __future__ import annotations

from typing import Any

from ..core.stage import Artifact, Stage, StageContext
from .train import resolve_train_run_id


class VisualizeStage(Stage):
    name = "visualize"
    depends_on = ("evaluate",)
    needs_gpu = True  # orbit video renders novel views; figures/metrics are CPU-only

    def _tr(self, ctx: StageContext) -> str:
        return resolve_train_run_id(ctx)

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("eval_json", ctx.layout.eval_json(self._tr(ctx)), "file")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        tr = self._tr(ctx)
        return [Artifact("metrics_summary",
                         ctx.layout.metrics_dir(tr) / "metrics_summary.json", "file")]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return {"train_run_id": self._tr(ctx), **ctx.config.report.model_dump()}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        from ..reporting import figures, metrics, videos

        cfg = ctx.config.report
        tr = self._tr(ctx)
        layout = ctx.layout
        log = ctx.logger
        fdir = layout.figures_dir(tr)
        vdir = layout.videos_dir(tr)
        produced: list[str] = []

        # --- metrics (always) ---
        js, md = metrics.write_summary_and_table(layout, tr)
        produced += ["metrics/metrics_summary.json", "metrics/metrics_table.md"]
        log.info("metrics summary -> %s", js)

        if not cfg.enabled:
            return {"figures": 0, "videos": 0, "metrics": True}

        # --- figures (F1-F4) ---
        renders_test = layout.renders_dir(tr) / "eval_test"
        ckpt = self._latest_ckpt(layout, tr)
        crop = self._crop_box(layout)
        fig_fns = {
            "training_curves": lambda: figures.training_curves(
                layout.metrics_jsonl(tr), fdir / "training_curves.png"),
            "qualitative": lambda: figures.qualitative_comparison(
                layout.eval_json(tr), renders_test, layout.colmap_images,
                fdir / "qualitative_test.png"),
            "gaussian_stats": lambda: figures.gaussian_stats(ckpt, fdir / "gaussian_stats.png")
                if ckpt else None,
            "per_view": lambda: figures.per_view_metrics(
                layout.eval_json(tr), fdir / "per_view_metrics.png"),
            "gaussian_centers": lambda: figures.gaussian_centers(
                ckpt, fdir / "gaussian_centers.png", crop) if ckpt else None,
        }
        n_fig = 0
        for name in cfg.figures:
            try:
                out = fig_fns[name]()
                if out:
                    produced.append(f"figures/{out.name}"); n_fig += 1
                    log.info("figure %s -> %s", name, out.name)
            except Exception as e:  # noqa: BLE001
                log.warning("figure %s failed: %s", name, e)

        # --- videos (V1 orbit, V3 progression) ---
        n_vid = 0
        renderer = None
        for name in cfg.videos:
            try:
                if name == "orbit":
                    renderer = renderer or self._renderer(ctx, tr)
                    if renderer is None:
                        continue
                    out = videos.orbit_video(
                        renderer, vdir / "orbit.mp4", n_frames=cfg.orbit_frames,
                        elevation_deg=cfg.orbit_elevation_deg,
                        radius_scale=cfg.orbit_radius_scale, fps=cfg.video_fps,
                        width=cfg.orbit_width, height=cfg.orbit_height)
                elif name == "progression":
                    out = videos.progression_video(
                        layout.training_dir(tr), vdir / "training_progression.mp4",
                        fps=cfg.progression_fps)
                else:
                    continue
                if out:
                    produced.append(f"videos/{out.name}"); n_vid += 1
                    log.info("video %s -> %s", name, out.name)
            except Exception as e:  # noqa: BLE001
                log.warning("video %s failed: %s", name, e)

        # --- assemble a combined report that embeds everything ---
        try:
            self._write_report(ctx, tr, produced)
        except Exception as e:  # noqa: BLE001
            log.warning("report assembly failed: %s", e)

        log.info("visualize: %d figures, %d videos, metrics ok", n_fig, n_vid)
        return {"figures": n_fig, "videos": n_vid, "n_artifacts": len(produced)}

    # ------------------------------------------------------------------ #
    def _latest_ckpt(self, layout, tr: str):
        from ..training.checkpoint import find_latest_valid
        return find_latest_valid(layout.checkpoints_dir(tr))

    def _crop_box(self, layout):
        import json
        tf = layout.normalize_transform
        if tf.exists():
            try:
                return json.loads(tf.read_text()).get("crop_box_normalized")
            except Exception:
                return None
        return None

    def _renderer(self, ctx: StageContext, tr: str):
        try:
            import torch
            if not torch.cuda.is_available():
                ctx.logger.warning("orbit video needs CUDA; skipping")
                return None
            from ..reporting.render import CheckpointRenderer
            near = 0.01
            if ctx.layout.normalize_transform.exists():
                import json
                near = max(1e-3, float(json.loads(ctx.layout.normalize_transform.read_text())
                                       .get("near", 0.01)) * 0.5)
            return CheckpointRenderer(ctx.layout, tr, ctx.config.train.sh_degree,
                                      device="cuda", near=near)
        except Exception as e:  # noqa: BLE001
            ctx.logger.warning("could not build renderer for orbit: %s", e)
            return None

    def _write_report(self, ctx: StageContext, tr: str, produced: list[str]) -> None:
        import json

        layout = ctx.layout
        rdir = layout.report_dir(tr)
        rdir.mkdir(parents=True, exist_ok=True)
        summ = {}
        js = layout.metrics_dir(tr) / "metrics_summary.json"
        if js.exists():
            summ = json.loads(js.read_text())
        figs = [p for p in produced if p.startswith("figures/")]
        vids = [p for p in produced if p.startswith("videos/")]
        lines = [f"# Report — {layout.dataset_id} / {tr}", ""]
        iq = summ.get("image_quality", {}).get("test", {})
        lines += ["## Held-out metrics (test)",
                  f"- PSNR **{iq.get('psnr')}**, SSIM **{iq.get('ssim')}**, "
                  f"LPIPS **{iq.get('lpips')}**, {iq.get('n_views')} views, "
                  f"{iq.get('render_fps')} FPS", ""]
        lines += ["## Figures", ""]
        for f in figs:
            lines.append(f"### {f}\n\n![{f}](../{f})\n")
        lines += ["## Videos", ""]
        for v in vids:
            lines.append(f"- [{v}](../{v})")
        lines += ["", "## Full metrics table", "",
                  f"See [metrics/metrics_table.md](../metrics/metrics_table.md)."]
        (rdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        # minimal self-contained HTML
        html = ["<!doctype html><meta charset='utf-8'><style>body{font-family:system-ui;"
                "max-width:1000px;margin:2rem auto}img{max-width:100%;border:1px solid #ddd}"
                "</style>", f"<h1>{layout.dataset_id} / {tr}</h1>",
                f"<p>test PSNR <b>{iq.get('psnr')}</b> · SSIM <b>{iq.get('ssim')}</b> · "
                f"LPIPS <b>{iq.get('lpips')}</b></p>"]
        for f in figs:
            html.append(f"<h3>{f}</h3><img src='../{f}'>")
        for v in vids:
            html.append(f"<h3>{v}</h3><video controls width='100%' src='../{v}'></video>")
        (rdir / "report.html").write_text("\n".join(html), encoding="utf-8")
