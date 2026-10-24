"""Invariant temperatures and the gate for the three Cu-Zr descriptions (SGTE Zeng 1994, PhaseForge MLIP-only,
PhaseForge+ tuned), from the TDB files alone (analytic model, scripts/model.py; no pycalphad needed).

Gate (pre-registered in orchid-phaseforge-demo PLAN.md, section 4): an invariant reaction with a printed ORCHID
temperature in the FIT set passes when the model has it and its temperature lies within 2u of the nearest printed
fit value (u = printed uncertainty, else 5 K). The printed values are inputs/printed_invariants.json.

IN-SAMPLE: the tuned TDB was fitted to these printed invariants (through their phase-equilibrium data), so its gate
count measures fit quality, not prediction. The paper's number is 9/13.

  python scripts/gate.py [OUT_DIR]        # writes OUT_DIR/gate.json (default: cached/)
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import model as m

ROOT = m.ROOT
TDBS = {"SGTE": ROOT / "tdb/CuZr-94Zen-LB.tdb", "MLIP-only": ROOT / "tdb/CuZr-PhaseForge-MLIP.tdb",
        "tuned": ROOT / "tdb/CuZr-PhaseForge-final.tdb"}


def main(out_dir):
    printed = json.load(open(ROOT / "inputs/printed_invariants.json"))
    P = {k: m.Params.from_tdb(v) for k, v in TDBS.items()}
    guess = {}
    for r in printed:
        if r["reaction"]:
            guess.setdefault(r["reaction"], r["T_K"])
    rows = []
    for rx in m.REACTIONS:
        name = rx[0]
        pf = [r for r in printed if r["reaction"] == name and r["split"] == "fit"]
        row = {"reaction": name, "printed_fit_K": sorted({r["T_K"] for r in pf}),
               "printed_holdout_K": sorted({r["T_K"] for r in printed if r["reaction"] == name and r["split"] == "holdout"})}
        for k, p in P.items():
            T = m.invariant_T(p, rx, guess.get(name, 1200.0), window=700)
            row[k] = None if T is None else round(T, 3)
            if pf:
                if T is None:
                    row[k + "_pass"] = False
                else:
                    best = min(pf, key=lambda r: abs(r["T_K"] - T))
                    row[k + "_pass"] = bool(abs(T - best["T_K"]) <= 2 * best["u_used_K"])
        rows.append(row)
    gated = [r for r in rows if r["printed_fit_K"]]
    counts = {k: [sum(r[k + "_pass"] for r in gated), len(gated)] for k in P}
    out = {"rule": "within 2u of the nearest printed FIT value (u printed, else 5 K); reaction must exist",
           "in_sample_note": "the tuned TDB was fitted to the fit-set phase data: its gate count is in-sample",
           "gate_counts": counts, "invariants": rows}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(out_dir) / "gate.json", "w"), indent=1)
    w = max(len(r["reaction"]) for r in rows)
    print(f"{'reaction':{w}s}  printed fit (K)            SGTE   MLIP-only  tuned")
    for r in rows:
        pf = ", ".join(f"{t:.0f}" for t in r["printed_fit_K"][:4]) + (" ..." if len(r["printed_fit_K"]) > 4 else "")
        cell = lambda k: ("   -  " if r[k] is None else f"{r[k]:6.0f}") + ("" if not r["printed_fit_K"] else (" ok" if r[k + "_pass"] else " x "))
        print(f"{r['reaction']:{w}s}  {pf:26s} {cell('SGTE')} {cell('MLIP-only')} {cell('tuned')}")
    print("invariants within the gate: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in counts.items()) + "  (tuned: in-sample)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "cached")
