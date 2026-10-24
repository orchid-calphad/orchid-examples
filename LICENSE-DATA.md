# Licences of the data in this repository

The **code** (all `.py` and `.sh` files, the notebook's code cells, the CI workflow) is MIT-licensed (`LICENSE`).
The **data** in this repository keep the licences of their sources:

| files | source | licence |
|---|---|---|
| values, points and datasets derived from ORCHID (`*/inputs/*.json`, `*/inputs/*.csv`, `quickstart/sample-release/`, cached results computed from them) | ORCHID data release (orchid-calphad/orchid-data; commit named in each example's README) | **CC BY 4.0**. Cite the ORCHID release and the original source of every value (each record carries its citation and DOI) |
| `tdb/CuZr-94Zen-LB.tdb`, `oneshot-benchmark/tdb/reference_panel/*.tdb` | B. Hallstedt, "The SGTE collection of binary datasets", Calphad 89 (2025) 102833, doi:10.1016/j.calphad.2025.102833, unchanged | **CC BY 4.0**. Cite Hallstedt (2025) and the original assessment named in each file's header |
| `CuZr-PhaseForge-*.tdb` | this work (PhaseForge / PhaseForge+ demonstrations); unary data and phase definitions copied from CuZr-94Zen-LB.tdb | CC BY 4.0 (review-only demonstration models, not assessments) |
| `cuzr-espei-mcmc/tdb/review_only/*.tdb`, `cuzr-espei-mcmc/tdb/CuZr-94Zen-LB_param.tdb` | refinements / parameterisations of CuZr-94Zen-LB.tdb (orchid-espei-demo) | CC BY 4.0 (derived from Hallstedt 2025); review-only demonstration models, not assessments |
| Materials Project and OQMD energies and structures (`mlip-formation-enthalpy/inputs/`, DFT columns) | Materials Project (Jain et al. 2013), OQMD (Saal et al. 2013; Kirklin et al. 2015) | **CC BY 4.0** |
| MLIP predictions (`mlip-formation-enthalpy/inputs/pairs_identical_set.csv`, `tdb/mlip_fit.json`) | outputs of the models named in `scripts/full/models.json`; some model weights are under the Academic Software License (non-commercial use of the weights; publication of results allowed) | CC BY 4.0 for the predictions as published here |

No publisher PDFs, page images, handbook drawings or handbook text are included. Points derived from Okamoto's
handbook diagrams appear only as the released coordinates of ORCHID Dataset 6 (assessed phase assemblages). ESPEI
dataset files are copied unchanged from the release; like the release, they may carry a short verbatim quote of the
source with its citation.
