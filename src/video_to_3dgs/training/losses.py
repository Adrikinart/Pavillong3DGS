"""Masked photometric (L1) + SSIM losses."""

from __future__ import annotations


def _gaussian_window(size: int, sigma: float, device, dtype):
    import torch
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = g[:, None] * g[None, :]
    return w[None, None]  # (1,1,H,W)


def ssim(pred, target, window_size: int = 11):
    """SSIM between two CHW images in [0,1]. Returns scalar (mean over channels)."""
    import torch
    import torch.nn.functional as F
    C, H, W = pred.shape
    dtype = pred.dtype
    win = _gaussian_window(window_size, 1.5, pred.device, dtype).repeat(C, 1, 1, 1)
    p = pred[None]
    t = target[None]
    pad = window_size // 2
    mu1 = F.conv2d(p, win, padding=pad, groups=C)
    mu2 = F.conv2d(t, win, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sig1 = F.conv2d(p * p, win, padding=pad, groups=C) - mu1_sq
    sig2 = F.conv2d(t * t, win, padding=pad, groups=C) - mu2_sq
    sig12 = F.conv2d(p * t, win, padding=pad, groups=C) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu12 + c1) * (2 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2))
    return s.mean()


def photometric_loss(render, gt, mask, l1_lambda: float, ssim_lambda: float):
    """render/gt: HWC in [0,1]. mask: HW1 in [0,1] or None. Returns (loss, l1, ssim_val)."""
    import torch
    if mask is not None:
        m = mask.clamp(0, 1)
        render = render * m
        gt = gt * m
        denom = m.sum().clamp_min(1.0)
        l1 = (torch.abs(render - gt).sum()) / (denom * render.shape[-1])
    else:
        l1 = torch.abs(render - gt).mean()
    # SSIM on CHW
    s = ssim(render.permute(2, 0, 1), gt.permute(2, 0, 1))
    loss = l1_lambda * l1 + ssim_lambda * (1.0 - s)
    return loss, l1.detach(), s.detach()


def psnr(render, gt, mask=None):
    import torch
    if mask is not None:
        m = mask.clamp(0, 1)
        diff = ((render - gt) ** 2) * m
        mse = diff.sum() / (m.sum().clamp_min(1.0) * render.shape[-1])
    else:
        mse = ((render - gt) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-10)))
