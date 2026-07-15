"""Generate a Markdown + HTML report for a training/evaluation run."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(ctx, train_run_id: str, eval_results: dict[str, Any]) -> Path:
    layout = ctx.layout
    rdir = layout.report_dir(train_run_id)
    rdir.mkdir(parents=True, exist_ok=True)

    lines = [f"# Report — {layout.dataset_id} / {train_run_id}", ""]
    model = eval_results.get("model", {})
    lines += [
        "## Model", "",
        f"- backend: `{ctx.config.train.backend}`",
        f"- gaussians: {model.get('n_gaussians')}",
        f"- checkpoint: `{eval_results.get('checkpoint')}` "
        f"({(model.get('checkpoint_bytes') or 0)/1e6:.1f} MB)",
        f"- peak VRAM: {(model.get('peak_vram_bytes') or 0)/1e9:.2f} GB",
        f"- step: {eval_results.get('step')}",
        "",
        "## Held-out metrics", "",
        "| split | views | PSNR | SSIM | LPIPS | render FPS |",
        "|---|---|---|---|---|---|",
    ]
    for split, res in eval_results.get("splits", {}).items():
        lines.append(f"| {split} | {res.get('n_views')} | {res.get('psnr')} | "
                     f"{res.get('ssim')} | {res.get('lpips')} | {res.get('render_fps')} |")
    lines += ["", "## Worst / best views", ""]
    for split, res in eval_results.get("splits", {}).items():
        pv = sorted([p for p in res.get("per_view", []) if p.get("psnr") is not None],
                    key=lambda x: x["psnr"])
        if pv:
            worst = ", ".join(f"{p['name']}({p['psnr']})" for p in pv[:3])
            best = ", ".join(f"{p['name']}({p['psnr']})" for p in pv[-3:])
            lines += [f"- **{split}** worst: {worst}", f"- **{split}** best: {best}"]
    lines += ["", "## Artifacts", "",
              f"- metrics: `{layout.metrics_jsonl(train_run_id)}`",
              f"- tensorboard: `{layout.tensorboard_dir(train_run_id)}`",
              f"- renders: `{layout.renders_dir(train_run_id)}`",
              f"- eval json: `{layout.eval_json(train_run_id)}`", ""]

    md = "\n".join(lines)
    (rdir / "report.md").write_text(md, encoding="utf-8")
    _write_html(rdir / "report.html", md)
    ctx.logger.info("report written to %s", rdir / "report.md")
    return rdir / "report.md"


def _write_html(path: Path, md: str) -> None:
    # minimal self-contained HTML (no external deps): wrap the markdown in <pre>
    esc = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = ("<!doctype html><meta charset='utf-8'>"
            "<style>body{font-family:system-ui,monospace;max-width:900px;margin:2rem auto;"
            "padding:0 1rem;line-height:1.5}pre{white-space:pre-wrap}</style>"
            f"<pre>{esc}</pre>")
    path.write_text(html, encoding="utf-8")
