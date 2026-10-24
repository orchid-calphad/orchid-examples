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
import argparse, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import model as m

ROOT = m.ROOT
NATOM = {"CU51ZR14": 65, "CU10ZR7": 17, "CUZR_B2": 2, "CUZR2": 3}
COMPOUND_OF = {"VV0004": "CU51ZR14", "VV0005": "CU10ZR7", "VV0006": "CUZR_B2", "VV0007": "CUZR2"}


def with_values(base, v):
    L, G = dict(base.L), dict(base.G)
    L[("LIQUID", 0)] = (v["VV0000"], v["VV0001"]); L[("LIQUID", 1)] = (v["VV0002"], v["VV0003"])
    for s, ph in COMPOUND_OF.items():
        G[ph] = (v[s], base.G[ph][1])
    return m.Params(L, G)


def gate(p, printed):
    guess = {}
    for r in printed:
        if r["reaction"]:
            guess.setdefault(r["reaction"], r["T_K"])
    rows, n = [], 0
    for rx in m.REACTIONS:
        pf = [r for r in printed if r["reaction"] == rx[0] and r["split"] == "fit"]
        T = m.invariant_T(p, rx, guess.get(rx[0], 1200.0), window=700)
        ok = bool(pf) and T is not None and abs(T - min(pf, key=lambda r: abs(r["T_K"] - T))["T_K"]) <= 2 * min(pf, key=lambda r: abs(r["T_K"] - T))["u_used_K"]
        rows.append({"reaction": rx[0], "T_K": None if T is None else round(float(T), 2), "printed_fit_K": sorted({r["T_K"] for r in pf}), "pass": bool(ok)})
        n += bool(pf)
    return rows, int(sum(r["pass"] for r in rows)), int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdb"); ap.add_argument("--trace"); ap.add_argument("-o")
    a = ap.parse_args()
    printed = json.load(open(ROOT / "inputs/printed_invariants.json"))
    base = m.Params.from_tdb(ROOT / "tdb/CuZr-94Zen-LB.tdb")
    syms = [p["symbol"] for p in json.load(open(ROOT / "tdb/params.json"))]
    models = {"SGTE start": base}
    if a.tdb:
        txt = Path(a.tdb).read_text()
        v = {s: float(x) for s, x in re.findall(r"FUNCTION\s+(VV\d+)\s+[\d.]+\s+([-+.\dEe]+)\s*;", txt)}
        models["ESPEI result (TDB)"] = with_values(base, v)
    if a.trace:
        t = np.load(a.trace)                       # (chains, iterations, parameters); unfinished iterations are 0
        it = int((np.abs(t).sum(axis=(0, 2)) > 0).sum())
        med = np.median(t[:, it // 2:it, :].reshape(-1, t.shape[2]), axis=0)
        models[f"posterior median ({it} iterations, second half)"] = with_values(base, dict(zip(syms, med)))
    out = {}
    for k, p in models.items():
        rows, ok, n = gate(p, printed)
        out[k] = {"gate_reactions_pass": ok, "gate_reactions": n, "invariants": rows}
        print(f"{k}: invariants within the gate {ok}/{n}")
        for r in rows:
            print(f"   {r['reaction']:36s} {('-' if r['T_K'] is None else format(r['T_K'], '.0f')):>6s} K  printed {r['printed_fit_K'][:4]}  {'ok' if r['pass'] else 'x'}")
    if a.o:
        json.dump(out, open(a.o, "w"), indent=1)


if __name__ == "__main__":
    main()
