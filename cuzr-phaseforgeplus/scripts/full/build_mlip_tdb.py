# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/build_mlip_tdb.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Pure-Python replacement for ATAT `sqs2tdb -fit` and `sqs2tdb -tdb` (PhaseForge sqscal steps 3-4), Cu-Zr.

sqs2tdb -fit: per lattice, fit the SQS energies E(x) (eV/atom) to a Redlich-Kister solution model with the terms of
  terms.in "1,0 / 2,1" (linear endmember term + binary interaction of degree 0 and 1, i.e. L0 and L1), by least squares
  (with 3 interior SQS points and 2 interaction parameters the fit is over-determined by one, as in sqs2tdb); results
  are written to <LAT>/fit_energy.out (x, E, E_fit, diff, name) like sqs2tdb.
sqs2tdb -tdb: assemble the TDB. As sqs2tdb does by default, endmember Gibbs energies of FCC_A1/BCC_A2/HCP_A3/LIQUID are
  the SGTE unary (Dinsdale 1991) lattice stabilities (here copied from CuZr-94Zen-LB.tdb, identical functions); the
  MLIP endmember energies only reference the mixing energies. Interaction parameters are written PhaseForge-style
  as L = VVa*1000 + VVb*T with VVa from the MLIP fit (kJ/mol) and VVb = 0 (no -vib/-sro, see PLAN.md).
Compounds (not handled by sqs2tdb/sqscal, which only treat lattices in ATAT's sqsdb): stoichiometric line compounds,
  G = a GHSERCU + b GHSERZR + VVa*1000 + VVb*T with VVa = (a+b) * MLIP formation energy at 0 K relative to the MLIP
  fcc-Cu and hcp-Zr (kJ/mol formula) and no T term (Neumann-Kopp; no vibrational or configurational entropy; not tuned).
LIQUID: L0, L1 from the MD energies at 2178 K (liquid_md.py), same fit as the solid lattices.
Phase names and sublattice models follow CuZr-94Zen-LB.tdb so that ORCHID datasets for that TDB apply unchanged.
Writes tdb/CuZr-PhaseForge-MLIP.tdb and tdb/mlip_fit.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import root, sgte_tdb

run = root / "runs/phaseforge"
sgte = sgte_tdb
ev = 96485.33212  # J/mol per eV
compounds = {"CU5ZR_C15B": (5, 1), "CU51ZR14": (51, 14), "CU8ZR3": (8, 3), "CU10ZR7": (10, 7), "CUZR_B2": (1, 1), "CUZR2": (1, 2)}
lats = ["LIQUID", "FCC_A1", "BCC_A2", "HCP_A3"]


def lattice_points(lat: str) -> list[dict]:
    """Load the SQS (or MD) energy points of one lattice, sorted by Zr fraction.

    Args:
        lat (str): Lattice name, e.g. FCC_A1 or LIQUID (LIQUID reads md.json instead of relax.json).

    Returns:
        One dict per point with keys x (Zr fraction), E (eV/atom), n (atoms), name, se, relax_strain and sg.

    Raises:
        SystemExit: If a point directory has no energy file.
    """
    pts = []
    for d in sorted((run / lat).glob("sqs_lev=*")):
        e = d / "energy"
        if not e.exists():
            raise SystemExit(f"missing energy in {d}")
        meta = json.loads((d / "sqs_meta.json").read_text())
        info = json.loads((d / ("md.json" if lat == "LIQUID" else "relax.json")).read_text())
        n = info["n_atoms"]
        pts.append(
            {
                "x": meta["x_Zr"],
                "E": float(e.read_text()) / n,
                "n": n,
                "name": d.name,
                "se": info.get("se_eV_atom_blocks_last_half"),
                "relax_strain": info.get("relax_strain"),
                "sg": info.get("sg_relaxed"),
            }
        )
    return sorted(pts, key=lambda p: p["x"])


def fit_rk(pts: list[dict]) -> dict:
    """Least-squares Redlich-Kister fit (L0, L1) of the mixing energies of one lattice.

    Args:
        pts (list[dict]): Points from ``lattice_points``, with keys x and E (eV/atom).

    Returns:
        Dict with L0 and L1 (J/mol-atom), the points x, the mixing energies dH_mix and fitted dH_fit (J/mol-atom), the
        RMS residual of the interior points (J/mol-atom), and the endmember energies E_endmembers_eV.
    """
    x = np.array([p["x"] for p in pts])
    energy = np.array([p["E"] for p in pts])
    e0 = energy[x == 0][0]
    e1 = energy[x == 1][0]
    dmix = (energy - (1 - x) * e0 - x * e1) * ev  # J/mol-atom
    inner = (x > 0) & (x < 1)
    design = np.column_stack([x * (1 - x), x * (1 - x) * ((1 - x) - x)])[inner]  # RK in (x_Cu - x_Zr)
    coef, *_ = np.linalg.lstsq(design, dmix[inner], rcond=None)
    fit = np.column_stack([x * (1 - x), x * (1 - x) * ((1 - x) - x)]) @ coef
    return {
        "L0": float(coef[0]),
        "L1": float(coef[1]),
        "x": x.tolist(),
        "dH_mix": dmix.tolist(),
        "dH_fit": fit.tolist(),
        "rms_resid": float(np.sqrt(np.mean((dmix - fit)[inner] ** 2))),
        "E_endmembers_eV": [float(e0), float(e1)],
    }


def main() -> None:
    """Fit every lattice, write the MLIP TDB and mlip_fit.json, and print the summary.

    Writes <run>/<LAT>/fit_energy.out per lattice, tdb/CuZr-PhaseForge-MLIP.tdb and tdb/mlip_fit.json under the work root.
    """
    out = {"lattices": {}, "compounds": {}}
    for lat in lats:
        pts = lattice_points(lat)
        f = fit_rk(pts)
        f["points"] = pts
        out["lattices"][lat] = f
        with (run / lat / "fit_energy.out").open("w") as fo:
            for p, dm, df in zip(pts, f["dH_mix"], f["dH_fit"], strict=True):
                fo.write(f"{1 - p['x']:g}\t{p['x']:g}\t\t{dm / ev:.5f}\t{df / ev:.5f}\t{(dm - df) / ev:.5f}\t\t{p['name']}\n")
    ecu = out["lattices"]["FCC_A1"]["E_endmembers_eV"][0]
    ezr = out["lattices"]["HCP_A3"]["E_endmembers_eV"][1]
    out["references_eV_atom"] = {"fcc_Cu": ecu, "hcp_Zr": ezr}
    # MLIP lattice stabilities (reported only; SGTE values used in the TDB)
    ls = {}
    for lat in ("FCC_A1", "BCC_A2", "HCP_A3"):
        e0, e1 = out["lattices"][lat]["E_endmembers_eV"]
        ls[lat] = {"Cu": (e0 - ecu) * ev, "Zr": (e1 - ezr) * ev}
    out["mlip_lattice_stabilities_J_mol"] = ls
    for ph, (a, b) in compounds.items():
        info = json.loads((run / "COMPOUNDS" / ph / "relax.json").read_text())
        n = info["n_atoms"]
        k = n // (a + b)
        e = info["energy_eV_cell"] / k  # eV per formula unit
        dh = (e - a * ecu - b * ezr) / (a + b) * ev  # J/mol-atom
        out["compounds"][ph] = {
            "dHf_J_mol_atom": dh,
            "G_const_J_mol_formula": dh * (a + b),
            "n_atoms_cell": n,
            "sg_relaxed": info["sg_relaxed"],
            "relax_strain": info["relax_strain"],
            "volume_A3_atom": info["volume_A3_atom"],
        }
    # ---- write the TDB -------------------------------------------------------------------------------------------
    src = sgte.read_text()
    head = src.split("$ Binary system")[0]
    head = head.replace(
        "$ Database for Cu-Zr from K.-J. Zeng 1994 et al. (Included in LB Vol. 3)",
        "$ Cu-Zr PhaseForge MLIP-informed database (GRACE-2L-OMAT; SQS by m_scraps); orchid-phaseforge-demo\n"
        "$ Unary data and phase/sublattice definitions copied from CuZr-94Zen-LB.tdb (SGTE, B. Hallstedt):",
    )
    vv, lines, k = [], [], 0
    syms = []

    def new_param(val: float, desc: str) -> str:
        """Allocate the next VV symbol and record its FUNCTION line and start value.

        Args:
            val (float): Start value of the parameter, in the unit given by ``desc``.
            desc (str): Description of the parameter for the symbol table.

        Returns:
            The allocated symbol name (VV0000, VV0001, ...).
        """
        nonlocal k
        name = f"VV{k:04d}"
        k += 1
        vv.append(f"FUNCTION {name} 298.15 {val:.6f}; 6000 N !")
        syms.append({"symbol": name, "start": val, "parameter": desc})
        return name

    for lat in lats:
        f = out["lattices"][lat]
        sub = "CU,ZR" if lat == "LIQUID" else "CU,ZR:VA"
        for j, key in enumerate(("L0", "L1")):
            a = new_param(f[key] / 1000.0, f"{key}({lat}) a [kJ/mol]")
            b = new_param(0.0, f"{key}({lat}) b [J/mol/K]")
            lines.append(f" PAR  L({lat},{sub};{j}) 298.15 {a}*1000+{b}*T; 6000 N PFMLIP !")
    for ph, (na, nb) in compounds.items():
        c = out["compounds"][ph]
        a = new_param(c["G_const_J_mol_formula"] / 1000.0, f"G({ph}) a [kJ/mol formula]")
        # no T term: the tunable set is PhaseForge's non-SGTE parameters plus linear-T terms of the solution-phase
        # interaction parameters only (PLAN.md amendment 1, paper configuration)
        lines.append(f" PAR  G({ph},CU:ZR) 298.15 {na}*GHSERCU+{nb}*GHSERZR+{a}*1000; 6000 N PFMLIP !")
    tdb = (
        head
        + "$ ------------------------------------------------------------------------------\n$ MLIP-informed binary parameters (VV symbols: PhaseForge/ESPEI convention)\n$\n"
        + "\n".join(vv)
        + "\n$\n"
        + "\n".join(lines)
        + "\n$\n LIST-OF-REFERENCE\n NUMBER  SOURCE\n  91Din  'A.T. Dinsdale, Calphad, 15, 317-425(1991).'\n"
        + "  PFMLIP 'PhaseForge workflow (Zhu, Sariturk, Arroyave 2025), GRACE-2L-OMAT energies, m_scraps SQS; this work'\n  !\n"
    )
    (root / "tdb").mkdir(exist_ok=True)
    (root / "tdb/CuZr-PhaseForge-MLIP.tdb").write_text(tdb)
    out["symbols"] = syms
    (root / "tdb/mlip_fit.json").write_text(json.dumps(out, indent=1))
    for lat in lats:
        f = out["lattices"][lat]
        print(f"{lat:<7} L0 {f['L0']:9.0f}  L1 {f['L1']:9.0f} J/mol  rms resid {f['rms_resid']:.0f} J/mol")
    for ph, c in out["compounds"].items():
        print(f"{ph:<11} dHf {c['dHf_J_mol_atom'] / 1000:7.2f} kJ/mol-atom")
    print("lattice stabilities (MLIP, J/mol):", {k: {e: round(v) for e, v in d.items()} for k, d in ls.items()})


if __name__ == "__main__":
    main()
