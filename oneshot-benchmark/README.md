# One-shot MLIP phase diagrams scored against ORCHID (Cu–Zr)

Can a machine-learned interatomic potential (MLIP), with no fitting to experiment, produce a usable phase diagram?
This example scores a "one-shot" CALPHAD description built from MLIP energies alone (PhaseForge, GRACE-2L-OMAT; see
`../cuzr-phaseforgeplus`) against ORCHID's phase evidence for Cu–Zr. Phase equilibrium is treated as a classification
problem: at every evidence point (T, x) the reference phase assemblage is compared with the model's, and the primary
metric is the **prevalence-weighted recall** of phase presence (each phase weighted by its support in the
evidence, coexistence counted). The track idea is J. Saal's; the metric is the PI's `phase_presence_metrics`.

![one-shot](figures/fig_cuzr_oneshot.png)

| model | field points (n = 28) | boundary points (n = 287) |
|---|---|---|
| **GRACE-2L-OMAT, MLIP-only (no fitting)** | **0.43 [0.30, 0.55]** | **0.40 [0.38, 0.43]** |
| SGTE CuZr-94Zen-LB (Zeng 1994), reference | 0.72 [0.59, 0.84] | 0.73 [0.70, 0.76] |
| PhaseForge+ final, **in-sample** (fitted to part of this evidence) | 0.76 [0.64, 0.87] | 0.66 [0.62, 0.69] |
| ten later Cu–Zr assessments (Hallstedt's SGTE collection), excluding Saunders 1985 | 0.70–0.98 | 0.62–0.78 |

95 % percentile bootstrap over points (2,000 resamples). Boundary points are correlated; cluster bootstraps by
equilibrium and by source are in each results JSON (GRACE boundary: [0.36, 0.44] and [0.34, 0.46]). Paired GRACE − SGTE:
field −0.30 [−0.43, −0.16], boundary −0.33 [−0.36, −0.30] (`results/comparison.json`). GRACE fails mainly on the
liquid (recall 0.02: the compounds and (βZr) are far too stable relative to the liquid), and B2 CuZr and Cu8Zr3 are
never stable. One system only: this tests the design of the track, it is not yet a benchmark.

**Other MLIPs: pending.** MLIP-only TDBs for CHGNet, MatterSim, ORB-v3 (two), MACE (two) and SevenNet-Omni, built with
the same protocol, are being computed (DGX Spark job); no scores are available yet, and none are reported here. When
the TDBs exist, they are scored with the command below and added to `results/`.

## Quick path: cached scores → figure; grid scoring (no pycalphad, about 30 s)

```sh
python -m venv .venv && . .venv/bin/activate      # CPython 3.12
pip install -r requirements.txt                  # numpy 2.3.2, scipy 1.16.1, matplotlib 3.10.7
sh run_quick.sh                                  # -> figures/fig_cuzr_oneshot.{png,pdf}, figures/grace_grid_score.json
```

`scripts/figure.py` draws the GRACE phase map (cached as `results/map_cuzr_grace.npz`, 4 K × 0.004 grid computed with
pycalphad) with the evidence points and the per-phase recall from `results/*.json`. The run then exercises the grid
path below on that cached map: it reproduces the TDB scores (0.426 / 0.401).

## Score your own model

**From a TDB** (full path; pycalphad):

```sh
pip install -r requirements-full.txt              # pycalphad 0.11.2 and its pinned dependencies (CPython 3.12)
python scripts/score.py my_model.tdb --label my_model [--names-as CuZr-94Zen-LB.tdb]
python scripts/compare.py                         # paired bootstraps of the three reference models
```

Phase names are mapped to ORCHID phase-registry ids through the registry's TDB names; a TDB the registry does not
know but which uses the phase names and sublattice models of a known file (as the PhaseForge TDBs use those of
CuZr-94Zen-LB) takes `--names-as`. Unknown phases become their own class (they can only add false positives).
Settings (same for every model): `pycalphad.equilibrium` at P = 101325 Pa, phases with NP > 1e-6, `pdens` 500; field
points one equilibrium; boundary points the union over a 5 × 5 band of T ± 10 K, x ± 0.01; failed cells re-solved
(drop-one-phase rule). About 15–60 s per model on one core.

**From an equilibrium grid** (any method: direct MLIP free energies, ML classifiers, digitised diagrams):

```sh
python scripts/score_grid.py my_grid.csv --label my_model     # columns T_K, x_Zr, phases ("LIQUID;CU10ZR7")
```

Phases are benchmark classes (`LIQUID, FCC_A1, BCC_A2, HCP_A3, C15B_CU5ZR, CU51ZR14, CU8ZR3, CU10ZR7, CUZR_B2, CUZR2,
CU24ZR13, CU2ZR, CU5ZR8, CU5ZR7`) or, with `--names-as FILE.tdb`, phase names of a registry TDB. Use at most
5 K × 0.005 over 700–1800 K; `scripts/npz_to_grid.py` writes the cached GRACE map in this format as a template.

## Evidence and release

ORCHID data: orchid-data commit **`c3d9781c26f72baaad64dd2b6aa6126ba487659e`** (release v2026.10, 2026-10-24).
`scripts/build_evidence.py` (full path; files fetched at that commit or from `ORCHID_DATA`) writes
`inputs/evidence.json`: 28 field points from Dataset 6 (assessed phase assemblages, one point inside each named field
of the assessed Cu–Zr diagram of Okamoto's handbook; released derived coordinates only) and 287 boundary points from
the literature ESPEI datasets (`espei/experimental/CU-ZR/*zpf*`, `espei/invariants/CU-ZR`; one equilibrium per ORCHID
observation id, each vertex a point). Points that were in the PhaseForge+ fit set are flagged from
`inputs/phaseforge_fit_split.json` (observation ids and field phase sets only). At this commit the evidence is
identical point for point to the source benchmark's (built at commit 0d9f7c4c3), and all 13 models rescored in the
pinned environment give identical per-point predictions (`scripts/score_all.sh`, 8 min in total on one core).

## Files

`tdb/`: CuZr-PhaseForge-MLIP.tdb (GRACE one-shot), CuZr-PhaseForge-final.tdb (PhaseForge+, in-sample),
CuZr-94Zen-LB.tdb and `reference_panel/` (ten later Cu–Zr assessments as transcribed in B. Hallstedt, Calphad 89 (2025)
102833, CC BY 4.0, unchanged; cite each original assessment named in the file header). `results/`: scores per model
(per-point predictions, settings, phase mapping, retries), `comparison.json`, the cached phase map. Scripts are
adapted from orchid-bench-pd (paths, release pinning, pickle-free cache, deterministic figure); `phase_metrics.py` is
the PI's function, copied verbatim with attribution.
