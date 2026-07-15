# Technical decisions

## Backend: direct gsplat (not Nerfstudio splatfacto)
Blackwell `sm_120` + `tiny-cuda-nn` is a known pain point; Nerfstudio drags it in.
Direct gsplat (which splatfacto uses under the hood anyway) gives clean Blackwell
support and full control over checkpointing/health/preemption/metrics. Built on
the *packaged* `gsplat.rasterization` + `DefaultStrategy` — NOT `examples/
simple_trainer.py` (not installed, no API stability). Splatfacto/orig-3dgs remain
future adapters behind the `TrainingBackend` ABC.

## Environment: fresh `v2gs` conda env on /home, built by local-disk mamba
Isolated from the user's existing gs_jepa env. Built with the fast local-disk
mamba (login-node NFS conda is a storm) but installed into an NFS prefix so the
healthy-NFS GPU nodes can import it. torch cu128 for sm_120; in-env `cuda-nvcc
12.8` so gsplat JIT-builds for Blackwell.

## SfM: COLMAP via CLI on node-local scratch; self-contained model readers
COLMAP's SQLite DB is hostile to the NFS mount → all COLMAP work runs on
`/var/tmp/$USER` and verified outputs are synced back. Shelling to the `colmap`
CLI works with any install. `.bin` model readers are self-contained (`colmap_io.py`)
to avoid a pycolmap wheel dependency. **CPU SIFT by default** (GPU SIFT on sm_120
is unverified); GPU behind a flag with an exhaustive-matcher fallback.

## Orchestration: filesystem as source of truth (no daemon/DB)
Per-stage status JSON is the completion marker; the runner writes RUNNING before
work and COMPLETED only after output validation, so a crash never yields a false
COMPLETED. Fingerprint = sha256(params + input checksums) drives skip/rerun and
downstream invalidation. Matches a single-user cluster tool and survives reboots.

## Primary target: Pavillon scene (per user)
Masks default OFF for the scene config; point-based normalization. The object and
turntable configs keep masking fully enabled/mandatory.

## Config: pydantic v2, frozen, layered YAML
defaults → cluster profile → user YAML → `--set` overrides → frozen
`config_resolved.yaml`. All models frozen for immutability (stable fingerprints).

## Scratch strategy: sync-back only after verification
`ScratchContext` deletes node-local scratch only after every output is sha256-
verified and atomically promoted; on failure it retains scratch (logged) for
debugging.

## Multi-GPU: parallel experiments, not distributed single-run
Object-scale scenes train on one GPU; sweeps fan out via Slurm job arrays sharing
one COLMAP build.
