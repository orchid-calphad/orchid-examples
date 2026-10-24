# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/liquid_md.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Liquid energies for the PhaseForge LIQUID fit (replaces sqscal's MLIPliquid/ternary_search block).

Protocol (MaterialsFramework Sqs2tdb, the Python successor of sqscal by the PhaseForge authors: NPT 3 ps then NVT
10 ps, energy = mean total energy of the last 20 % of NVT), with two documented changes:
  * one temperature for all compositions, T = Tm(Zr) + 50 K = 2178 K (sqscal: composition-averaged Tm + 50 K per
    structure; a common T makes the mixing energy an enthalpy of mixing at fixed T and cancels kinetic energy);
  * a 2 ps NVT pre-melt at 3500 K from the fcc SQS start, so every composition is liquid before equilibration.
Start: 108-atom fcc SQS (sqs_mscraps.py LIQUID), dt = 1 fs, NPT at 1 atm (MaterialsFramework npt_nose_hoover).
Writes energy (eV/cell), CONTCAR, md.json (per-atom energies, block-average standard error, mean T, volume, MSD).
Usage: python liquid_md.py <dir>
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid  # noqa: E402,F401
import json, sys, time
from pathlib import Path
import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from materialsframework.calculators import GraceCalculator

T, TMELT = 2178.0, 3500.0
d = Path(sys.argv[1])
steps = dict(melt=2000, npt=3000, nvt=10000)
if len(sys.argv) > 2 and sys.argv[2] == "--short":   # smoke test
    steps = dict(melt=50, npt=50, nvt=100)
a = read(d / "POSCAR", format="vasp")
t0 = time.time()
c = GraceCalculator(model="GRACE-2L-OMAT", temperature=TMELT, timestep=1.0, ensemble="nvt_nose_hoover", loginterval=500, logfile=str(d / "md.log"))
r = c.run(a, steps=steps["melt"])
c.temperature = T; c.ensemble = "npt_nose_hoover"
r = c.run(r["final_structure"], steps=steps["npt"])
s_npt = r["final_structure"]
c.ensemble = "nvt_nose_hoover"
r = c.run(s_npt, steps=steps["nvt"])
n = len(a)
etot = np.asarray(r["total_energy"]); epot = np.asarray(r["potential_energy"]); temp = np.asarray(r["temperature"])
last = max(1, int(0.2 * len(etot)))
E = float(np.mean(etot[-last:]))
half = etot[len(etot) // 2:]
blocks = np.array_split(half, 10)
se = float(np.std([b.mean() for b in blocks], ddof=1) / np.sqrt(len(blocks)))
pos = np.array([np.asarray(v) for v in []])
(d / "energy").write_text("%.8f\n" % E)
r["final_structure"].to(filename=str(d / "CONTCAR"), fmt="poscar")
# diffusion check on the NVT segment: unwrapped displacement estimate from velocities (sum v dt)
vel = np.asarray(r["velocities"])           # (steps, n, 3) in ASE units (A / (A*sqrt(u/eV)))
from ase import units
disp = vel.sum(axis=0) * 1.0 * units.fs * c.interval if vel.ndim == 3 else np.zeros((n, 3))
msd = float(np.mean(np.sum(disp ** 2, axis=1)))
info = {"T_K": T, "T_premelt_K": TMELT, "steps": steps, "n_atoms": n, "formula": a.get_chemical_formula(),
        "energy_eV_cell": E, "energy_eV_atom": E / n, "epot_eV_atom_last20": float(np.mean(epot[-last:]) / n),
        "se_eV_atom_blocks_last_half": se / n, "T_mean_last20": float(np.mean(temp[-last:])),
        "volume_A3_atom_npt_end": s_npt.volume / n, "msd_nvt_A2_estimate": msd, "seconds": round(time.time() - t0, 1),
        "model": "GRACE-2L-OMAT"}
(d / "md.json").write_text(json.dumps(info, indent=1))
if (d / "wait").exists(): (d / "wait").unlink()
print(json.dumps(info))
