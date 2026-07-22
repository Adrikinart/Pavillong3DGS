# Pipeline configs

## Layout

```
configs/pipeline/
  templates/     reusable archetypes  (object_default, object_high_quality,
                                        object_turntable, smoke_test)
  pavillon/      the Pavillon scene    (single-sided carved wall panel)
  casque/        the Casque helmet     (object-centric orbit, multi-camera)
```

**One directory per capture.** A new object gets its own folder with one canonical
config; that keeps runs, docs and results grouped by subject.

## How a config is resolved

`load_config` merges, in order: packaged defaults → cluster profile → the user YAML
you pass → `--set key=value` overrides. Every config here is therefore *standalone*
— it only needs to state what differs from the defaults — and configs do **not**
include one another.

## One config per object, `--set` for experiments

A single-parameter experiment does **not** need a new file. Sweep with `--set`:

```bash
# capacity sweep on one config, no new files
for cap in 190000 375000 750000; do
  sbatch scripts/slurm/train.sbatch configs/pipeline/casque/casque.yaml \
         --force --from-stage train --set train.densification.cap_max=$cap \
                                     --set train.train_run_id=casque_cap${cap}
done
```

Write a **new** config only when several *coupled* parameters change together and the
combination is worth naming — e.g. a different backend (`casque_2dgs.yaml`: backend +
its regularisers + densification all change at once). The `pavillon/` folder still
carries ~13 near-duplicate sweep files: that is the historical anti-pattern this
convention exists to avoid. Prefer `--set`.

## Reproducing a specific result

Each row of a `docs/reproduce_*.md` table and each `experiments/registry.csv` row names
the config (and any `--set` overrides) that produced it. Training a sibling model on an
existing dataset reuses its SfM: add `--force --from-stage train`.
