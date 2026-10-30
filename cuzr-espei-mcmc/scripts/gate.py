"""Invariant temperatures and the gate of the paper's Cu-Zr examples for the SGTE start and an ESPEI result.

The 8 fitted coefficients (tdb/params.json) are read either from an ESPEI output TDB (its FUNCTION VV#### lines) or as
the posterior median of an ESPEI trace (second half of the iterations, all chains), and placed into the SGTE
description; invariants are found with the analytic model (scripts/model.py, as cuzr-phaseforgeplus). Gate: every
reaction with a printed ORCHID fit-set temperature lies within 2u of the nearest printed value (u printed, else 5 K)
and exists in the model (inputs/printed_invariants.json, release 778cc6bfa).

  python scripts/gate.py                                   # SGTE start only
  python scripts/gate.py --tdb work/refined_REVIEW_ONLY.tdb
  python scripts/gate.py --trace work/trace.npy [--lnprob work/lnprob.npy]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import model as m  # noqa: E402

root = m.root
natom = {"CU51ZR14": 65, "CU10ZR7": 17, "CUZR_B2": 2, "CUZR2": 3}  # atoms per formula unit
compound_of = {"VV0004": "CU51ZR14", "VV0005": "CU10ZR7", "VV0006": "CUZR_B2", "VV0007": "CUZR2"}


def with_values(base: m.Params, v: dict[str, float]) -> m.Params:
    """Parameter set of base with the fitted coefficients replaced by the values of v.

    Args:
        base (m.Params): Parameter set supplying the unchanged coefficients (the SGTE values).
        v (dict[str, float]): VV symbol to fitted value; liquid terms take VV0000 to VV0003, compounds VV0004 to VV0007.

    Returns:
        m.Params: New parameter set; base is not modified.
    """
    liquid, compound = dict(base.liquid), dict(base.compound)
    liquid[("LIQUID", 0)] = (v["VV0000"], v["VV0001"])
    liquid[("LIQUID", 1)] = (v["VV0002"], v["VV0003"])
    for s, ph in compound_of.items():
        compound[ph] = (v[s], base.compound[ph][1])
    return m.Params(liquid, compound)


def gate(p: m.Params, printed: list[dict]) -> tuple[list[dict], int, int]:
    """Check every model invariant against the printed ORCHID invariants.

    Args:
        p (m.Params): Parameter set to evaluate.
        printed (list[dict]): Printed invariant records (reaction, T_K, split, u_used_K).

    Returns:
        tuple[list[dict], int, int]: Per-reaction rows (reaction, model T_K, printed fit T_K values, pass flag), the
        number of rows passing, and the number of reactions with a printed fit-set value.
    """
    guess = {}
    for r in printed:
        if r["reaction"]:
            guess.setdefault(r["reaction"], r["T_K"])
    rows, n = [], 0
    for rx in m.reactions:
        pf = [r for r in printed if r["reaction"] == rx[0] and r["split"] == "fit"]
        temp = m.invariant_temperature(p, rx, guess.get(rx[0], 1200.0), window=700)
        if pf and temp is not None:
            nearest = min(pf, key=lambda r: abs(r["T_K"] - temp))
            ok = abs(temp - nearest["T_K"]) <= 2 * nearest["u_used_K"]
        else:
            ok = False
        rows.append(
            {
                "reaction": rx[0],
                "T_K": None if temp is None else round(float(temp), 2),
                "printed_fit_K": sorted({r["T_K"] for r in pf}),
                "pass": bool(ok),
            }
        )
        n += bool(pf)
    return rows, int(sum(r["pass"] for r in rows)), int(n)


def main() -> None:
    """Run the gate for the SGTE start and for any ESPEI result given, print the table and optionally write JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdb")
    ap.add_argument("--trace")
    ap.add_argument("-o")
    a = ap.parse_args()
    printed = json.loads((root / "inputs/printed_invariants.json").read_text())
    base = m.Params.from_tdb(root / "tdb/CuZr-94Zen-LB.tdb")
    syms = [p["symbol"] for p in json.loads((root / "tdb/params.json").read_text())]
    models = {"SGTE start": base}
    if a.tdb:
        txt = Path(a.tdb).read_text()
        v = {s: float(x) for s, x in re.findall(r"FUNCTION\s+(VV\d+)\s+[\d.]+\s+([-+.\dEe]+)\s*;", txt)}
        models["ESPEI result (TDB)"] = with_values(base, v)
    if a.trace:
        t = np.load(a.trace)  # (chains, iterations, parameters); unfinished iterations are 0
        it = int((np.abs(t).sum(axis=(0, 2)) > 0).sum())
        med = np.median(t[:, it // 2 : it, :].reshape(-1, t.shape[2]), axis=0)
        models[f"posterior median ({it} iterations, second half)"] = with_values(base, dict(zip(syms, med, strict=True)))
    out = {}
    for k, p in models.items():
        rows, ok, n = gate(p, printed)
        out[k] = {"gate_reactions_pass": ok, "gate_reactions": n, "invariants": rows}
        print(f"{k}: invariants within the gate {ok}/{n}")
        for r in rows:
            tk = "-" if r["T_K"] is None else format(r["T_K"], ".0f")
            mark = "ok" if r["pass"] else "x"
            print(f"   {r['reaction']:36s} {tk:>6s} K  printed {r['printed_fit_K'][:4]}  {mark}")
    if a.o:
        with Path(a.o).open("w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
