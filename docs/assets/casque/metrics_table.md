# Metrics — casque_orbit_07ccd886 / casque_gsplat

## Held-out image quality (test)

| metric | value | std | min | max |
|---|---|---|---|---|
| PSNR | 20.3531 | 2.2525 | 14.337 | 23.026 |
| SSIM | 0.8636 | 0.046 | 0.7823 | 0.9273 |
| LPIPS | 0.271 | 0.0861 | 0.1677 | 0.5108 |
| render FPS | 1.33 | | | |
| # views | 13 | | | |

## Efficiency

- # Gaussians: 1301628
- checkpoint: 1015.19 MB
- peak VRAM: 2.203 GB

## Reconstruction (COLMAP)

- registered: 134/136 (ratio 0.9852941176470589)
- mean reproj error: 0.9738462233568427 px
- points: 35714, mean track length: 5.215293722349779

## Training

- status: COMPLETED, final step: 29999
- best val PSNR: 20.421
- duration: 1968.54 s

## Provenance

- git: `9082abd42b89de473f55346a67bf28532a9b7734` | torch 2.13.0+cu130 (cuda 13.0) | GPU NVIDIA RTX PRO 4500 Blackwell
