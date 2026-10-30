"""Invariant temperatures and the gate for the three Cu-Zr descriptions.

The descriptions are SGTE Zeng 1994, PhaseForge MLIP-only and PhaseForge+ tuned. They are evaluated from the TDB files
alone (analytic model, scripts/model.py; no pycalphad needed).

Gate (pre-registered in orchid-phaseforge-demo PLAN.md, section 4): an invariant reaction with a printed ORCHID
temperature in the FIT set passes when the model has it and its temperature lies within 2u of the nearest printed
fit value (u = printed uncertainty, else 5 K). The printed values are inputs/printed_invariants.json.

IN-SAMPLE: the tuned TDB was fitted to these printed invariants (through their phase-equilibrium data), so its gate
count measures fit quality, not prediction. The paper's number is 9/13.

  python scripts/gate.py [OUT_DIR]        # writes OUT_DIR/gate.json (default: cached/)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # noqa: E402 - model.py sits beside this script
import model as m  # noqa: E402

root = m.root
tdbs = {
    "SGTE": root / "tdb/CuZr-94Zen-LB.tdb",
    "MLIP-only": root / "tdb/CuZr-PhaseForge-MLIP.tdb",
    "tuned": root / "tdb/CuZr-PhaseForge-final.tdb",
}
default_guess_k = 1200.0  # K, search centre when no printed value of the reaction exists
search_window_k = 700  # K, half-width of the invariant search


def _table_cell(row: dict, key: str) -> str:
    """Format one model cell of the printed table: temperature, then ok or x when a fit value is gated.

    Args:
        row (dict): One invariant row produced by main().
        key (str): Description name, "SGTE", "MLIP-only" or "tuned".

    Returns:
        str: Fixed-width temperature in K ("-" when the model has no invariant), with " ok", " x " or nothing.
    """
    value = "   -  " if row[key] is None else f"{row[key]:6.0f}"
    return value + ("" if not row["printed_fit_K"] else (" ok" if row[key + "_pass"] else " x "))


def main(out_dir) -> None:
    """Compute the invariant table for the three descriptions, write gate.json and print the table.

    Args:
        out_dir (str | Path): Directory that receives gate.json (created if missing).
    """
    printed = json.loads((root / "inputs/printed_invariants.json").read_text(encoding="utf-8"))
    params = {k: m.Params.from_tdb(v) for k, v in tdbs.items()}
    guess = {}
    for r in printed:
        if r["reaction"]:
            guess.setdefault(r["reaction"], r["T_K"])
    rows = []
    for rx in m.reactions:
        name = rx[0]
        pf = [r for r in printed if r["reaction"] == name and r["split"] == "fit"]
        row = {
            "reaction": name,
            "printed_fit_K": sorted({r["T_K"] for r in pf}),
            "printed_holdout_K": sorted({r["T_K"] for r in printed if r["reaction"] == name and r["split"] == "holdout"}),
        }
        for k, p in params.items():
            temperature = m.invariant_temperature(p, rx, guess.get(name, default_guess_k), window=search_window_k)
            row[k] = None if temperature is None else round(temperature, 3)
            if pf:
                if temperature is None:
                    row[k + "_pass"] = False
                else:
                    best = min(pf, key=lambda r: abs(r["T_K"] - temperature))
                    row[k + "_pass"] = bool(abs(temperature - best["T_K"]) <= 2 * best["u_used_K"])
        rows.append(row)
    gated = [r for r in rows if r["printed_fit_K"]]
    counts = {k: [sum(r[k + "_pass"] for r in gated), len(gated)] for k in params}
    out = {
        "rule": "within 2u of the nearest printed FIT value (u printed, else 5 K); reaction must exist",
        "in_sample_note": "the tuned TDB was fitted to the fit-set phase data: its gate count is in-sample",
        "gate_counts": counts,
        "invariants": rows,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "gate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    w = max(len(r["reaction"]) for r in rows)
    print(f"{'reaction':{w}s}  printed fit (K)            SGTE   MLIP-only  tuned")
    for r in rows:
        fit_text = ", ".join(f"{t:.0f}" for t in r["printed_fit_K"][:4]) + (" ..." if len(r["printed_fit_K"]) > 4 else "")
        print(
            f"{r['reaction']:{w}s}  {fit_text:26s} {_table_cell(r, 'SGTE')} {_table_cell(r, 'MLIP-only')} {_table_cell(r, 'tuned')}"
        )
    print("invariants within the gate: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in counts.items()) + "  (tuned: in-sample)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else root / "cached")
