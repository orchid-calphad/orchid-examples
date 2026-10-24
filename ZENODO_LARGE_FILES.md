# Large outputs to archive on Zenodo (not in this repository)

The examples commit only small inputs, cached summaries and figures (the repository stays under 100 MB). The bulky
outputs of the full paths below are to be archived as a Zenodo record linked to this repository (sizes as of
2026-10-24, from the source working repositories).

| example | output | source | size | notes |
|---|---|---|---:|---|
| cuzr-phaseforgeplus | MLIP runs: SQS and compound relaxations, liquid MD logs and final structures (`runs/phaseforge/`) | orchid-phaseforge-demo | 14 MB | inputs of `build_mlip_tdb.py`; lets the TDB be rebuilt without the MLIP |
| cuzr-phaseforgeplus | all fit runs and optimizer traces (pre-registered, PR #268 reruns, Amendment 4 variants, final refit), dense-variant traces | orchid-phaseforge-demo `runs/` | 13 MB | only the selected final start is cached here |
| cuzr-phaseforgeplus | NUTS chains (exploration and UQ draws, tuning traces) | orchid-phaseforge-demo `runs/hmc/` | 1.8 MB | summary in `cached/hmc/` |
| cuzr-espei-mcmc | ESPEI MCMC chains (`trace.npy`, `lnprob.npy`) and logs of v1 (main, perturbed, run A) and v2 (ensembles A, B); gradient-fit outputs of v3/v4 | orchid-espei-demo `runs/` | 90 MB | review-only; refined TDBs are not to be redistributed as assessments |
| cuzr-espei-mcmc | posterior and Laplace draw tables, phase-diagram line caches | orchid-espei-demo `results/` | 47 MB | |
| mlip-formation-enthalpy | raw per-structure MLIP outputs (8 models × 926 structures, relaxed and single-point energies, flags, timings) | orchid-mlip-benchmark `results/spark_20261024/`, `results/studio_addendum_r2/` | 33 MB | inputs of `scripts/full/analyze_raw.py` |
| mlip-formation-enthalpy | full calibration output (`CALIBRATION_MLIP.json` with leave-system-out tables, per-model record sets) | orchid-mlip-benchmark `results/full_analysis/` | 2.4 MB | a trimmed copy is in `cached/` |
| oneshot-benchmark | MLIP-only TDBs of the seven further MLIPs and their build outputs | pending (DGX Spark job) | – | to be added when the builds finish |

Not archived anywhere public: the dense assessed-field ZPF grid of the PhaseForge+ final fit (derived from a handbook
drawing, not part of the ORCHID release; available on request).
