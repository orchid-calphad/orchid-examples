# ORCHID examples

Runnable, documented examples of the workflows described in the ORCHID paper. Each folder rebuilds its figures in
minutes, without a GPU, from cached outputs (the **quick path**), and documents the **full path** that produced those
outputs, with its compute requirements, timings, pinned environment and pinned ORCHID release commit.

ORCHID (Open Repository for CALPHAD Hypotheses, Intelligence, and Discovery): data at
[orchid-calphad/orchid-data](https://github.com/orchid-calphad/orchid-data) / <https://data.orchid-project.org>,
project site <https://orchid-project.org>. Paper: *TODO: arXiv / journal link*. Archive of this repository: Zenodo
[doi:10.5281/zenodo.23736043](https://doi.org/10.5281/zenodo.23736043) (placeholder until the DOI is reserved). Values in the ORCHID release are **Version 1, machine-extracted**.

| example | what it shows | headline result | quick path |
|---|---|---|---|
| [`cuzr-phaseforgeplus/`](cuzr-phaseforgeplus/) | MLIP → CALPHAD → gradient tuning against ORCHID data: a Cu–Zr description built with PhaseForge from GRACE-2L-OMAT energies alone, tuned with PhaseForge+ (Kunselman's gradient ESPEI); the paper's three-diagram overlay | invariants within the gate: MLIP-only 0/13, tuned **9/13 (in-sample, post hoc)**, SGTE 12/13 | ~4 min |
| [`cuzr-espei-mcmc/`](cuzr-espei-mcmc/) | the classic ESPEI route: `load_datasets`, `apply_tags`, `mcmc` on the SGTE Cu–Zr description | **negative example**: no variant passed the gate (MCMC not converged; best gradient variant 10/13; fallback m7 8/13): refining the 1994 assessment improves activities but cannot keep the invariants — a model-adequacy result | ~1 min |
| [`oneshot-benchmark/`](oneshot-benchmark/) | prevalence-weighted recall of a one-shot MLIP phase diagram against ORCHID phase evidence; score your own model (TDB or equilibrium grid) | GRACE one-shot 0.43 / 0.40 (field / boundary) vs SGTE 0.72 / 0.73; seven more MLIPs pending | ~1 min |
| [`mlip-formation-enthalpy/`](mlip-formation-enthalpy/) | eight MLIPs vs ORCHID experimental formation enthalpies, with the rules of ORCHID's DFT benchmark | saturation: σ 8.6–9.7 kJ/mol-atom vs MP 8.8, OQMD 7.5; σ_MLIP − σ_MP CI contains 0 for every model; r(MLIP, MP residuals) 0.96–0.997 | ~1 min |
| [`quickstart/`](quickstart/) | executed notebook: query a system from the Parquet release, filter (accepted, evidence class, reader agreement), experiment vs DFT, export ESPEI datasets | – | ~20 s, offline |

## Run an example

```sh
git clone https://github.com/orchid-calphad/orchid-examples && cd orchid-examples/<example>
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # quick path: exact versions
sh run_quick.sh                           # regenerates figures/ (and the cached tables) from cached outputs
```

`python tests/check_quick_paths.py [example ...]` runs quick paths into a temporary directory and checks that every
output regenerates byte-identically (sha256) against the committed files; `python tests/check_hygiene.py` checks
the repository rules (no absolute local paths, no file over 50 MB, under 100 MB in total). The CI workflow
(`.github/workflows/examples.yml`) runs both for every example; on Linux it allows a small pixel difference for PNGs
(`--pixel-tol`), because rasterisation can differ across platforms.

**ORCHID data access.** Quick paths never download anything. Full paths and the quickstart read release files through
`common/orchid_data.py`: `ORCHID_DATA=<local orchid-data checkout or unpacked Zenodo archive>` (check it out at the
commit pinned in the example's README), else a download of the commit-pinned file from GitHub, cached in
`~/.cache/orchid-examples` (`ORCHID_DATA_URL` overrides the URL template; `https://data.orchid-project.org/{path}`
serves the latest release).

## Credits

- **Courtney Kunselman** — the analytic-gradient ESPEI branch (github.com/cjkunselman18/ESPEI, branch
  `Gradient-based_update`, PhasesResearchLab/ESPEI pull request #268, unmerged; pinned here at
  `9039ea215c8946ff6761528071db9e519f37dfb1` for the fit and `4888897eec880b6b9a67484dfb680d7de0a00750` as locked
  by PhaseForgePlus) and the gradient-informed HMC (NUTS) protocol, with her littlemcmc fork
  (github.com/cjkunselman18/littlemcmc @ `2917beed02f7ddee7258aa7b987e87c51391b584`). Kunselman, Bocklund, Otis,
  Arróyave, Acta Mater. 300 (2025) 121453 (doi:10.1016/j.actamat.2025.121453) and Acta Mater. 315 (2026) 122412
  (doi:10.1016/j.actamat.2026.122412); Kunselman, Zhu, Sarıtürk, Arróyave, J. Phase Equilib. Diffus. (2025),
  doi:10.1007/s11669-025-01222-2.
- **Brandon Bocklund and the ESPEI developers** — ESPEI (Bocklund et al., MRS Commun. 9 (2019) 618,
  doi:10.1557/mrc.2019.59), github.com/PhasesResearchLab/ESPEI; **pycalphad** (Otis and Liu, J. Open Res. Softw. 5 (2017)
  1, doi:10.5334/jors.140).
- **PhaseForge** (Sarıtürk, Zhu, Arróyave; doi:10.5281/zenodo.15730912; Zhu, Sarıtürk, Arróyave, npj Comput. Mater.
  11 (2025) 340 and Acta Mater. 286 (2025) 120747), **PhaseForgePlus** (Sarıtürk, Kunselman, Zhu;
  doi:10.5281/zenodo.17519087) and **MaterialsFramework** (Sarıtürk, GPL-3.0).
- **m_scraps** (Arróyave, github.com/rarroyave/m_scraps, extending SCRAPS of Singh et al., Nat. Comput. Sci. 1 (2021) 54).
- MLIPs: GRACE (Bochkarev, Lysogorskiy, Drautz, Phys. Rev. X 14 (2024) 021036) and the models named in
  `mlip-formation-enthalpy/`; SGTE binary collection (Hallstedt, Calphad 89 (2025) 102833, CC BY 4.0); Materials
  Project (Jain et al. 2013) and OQMD (Saal et al. 2013; Kirklin et al. 2015).
- The one-shot phase-diagram track is James Saal's idea.

None of these packages' code is copied here; every dependency is pinned by repository URL and commit (or PyPI
version) in the examples' requirement files. All are public at the pinned commits (checked 2026-10-24).

## Licence

Code: MIT (`LICENSE`). Data derived from ORCHID: CC BY 4.0; other data keep their sources' licences
(`LICENSE-DATA.md`). Contributing: `CONTRIBUTING.md` (data corrections go to orchid-data). Large outputs:
`ZENODO_LARGE_FILES.md`. Citation: `CITATION.cff`.

## Funding

This work was supported by the U.S. National Science Foundation under awards [OAC-2608825](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2608825) (CSSI Elements: PhaseForge) and [2534344](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2534344) (Autonomous Robotic Metallurgist: An NSF Materials Innovation Platform, ARM-MIP).

## Software and data credits

See [papers and repositories](CITATIONS.md) for pycalphad, ESPEI, PhaseForge, PhaseForge+, Materials Project, and OQMD. [Machine-readable citations](CITATIONS.json) are also provided.
