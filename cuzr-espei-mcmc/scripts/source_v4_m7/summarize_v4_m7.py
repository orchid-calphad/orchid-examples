"""v4 m7 post-hoc summary (Amendment 4 fallback, model-adequacy test).

m6 plus third liquid term 2L (a only, start 0, bound +-40 kJ/mol), run at ZPF factor x30 (the v4 factor whose optimum
fails the fewest gate reactions: 3 of 13), starts SGTE, P2, P4.

Same data (data/v4_m7 -> data/v4_w30), hold-out, gate and metric functions as summarize_v4.py (analyze_v4.gate, .metrics,
.inv_job). Writes results/v4_m7/summary_v4_m7.{json,md} and tdb/v4/CuZr-94Zen-LB_ORCHID-v4_m7_REVIEW-ONLY.tdb.
No Laplace / Hessian (as summarize_v4.py).

Reference copy: analyze_v4, make_start_tdb_v3 and the v2/v3 folders are not shipped in this example; the script is
kept as the record of how the m7 result was produced, not as runnable code of this example.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["V4_MODEL"] = "m7"
os.environ["V4_TAG"] = "v4_m7"
here = Path(__file__).absolute().parent
sys.path[:0] = [str(here), str(here / ".." / "v3"), str(here / ".." / "v2")]
import analyze_v4 as analyze  # noqa: E402
import model_v4 as mst  # noqa: E402

assert mst.model_kind == "m7"
root = mst.root
tag = "v4_m7"
analyze.TAG = tag  # attribute of the analysis module (external name, kept)
key_reactions = ["L = CuZr (congruent)", "L = Cu10Zr7 (congruent)"]


def _mae(invd: dict, split: str) -> float | None:
    """Mean absolute error of the model invariant temperatures against the printed ones of one split.

    Args:
        invd (dict): Reaction name to model invariant temperature in K.
        split (str): Printed split, "fit" or "holdout".

    Returns:
        float | None: Mean absolute error in K, or None when no printed value of the split has a model value.
    """
    e = [
        abs(invd[r["reaction"]] - r["T_K"])
        for r in analyze.PRINTED
        if r["split"] == split and r["reaction"] and invd.get(r["reaction"]) is not None
    ]
    return float(np.mean(e)) if e else None


def _rmse(metrics: dict, k: str, s: str) -> float | None:
    """RMSE of one metric for one parameter set, or None when absent.

    Args:
        metrics (dict): Output of analyze.metrics.
        k (str): Metric name, e.g. "fit/ACR".
        s (str): Parameter set, "start" or "optimum".

    Returns:
        float | None: The RMSE in the metric's unit.
    """
    return metrics.get(k, {}).get(s, {}).get("rmse")


def _first_trace_line(path: str) -> dict:
    """Parse the first JSON line of a trace file.

    Args:
        path (str): Trace file path.

    Returns:
        dict: The parsed first record.
    """
    with Path(path).open(encoding="utf-8") as fh:
        return json.loads(fh.readline())


def _table(row: dict, m6: dict, out: dict) -> list[str]:
    """Markdown lines of the comparison table and its footnotes.

    Args:
        row (dict): Summary record of the m7 optimum (best start, optimum vector, gate detail, RMSEs).
        m6 (dict): Summary record of the m6 x30 variant from results/v4/summary_v4.json.
        out (dict): Full summary (start block, pass criterion and printed values).

    Returns:
        list[str]: Table lines, then the footnote lines.
    """
    lines = [
        "| model | ACR RMSE fit (kJ/mol) | ACR RMSE held-out | ZPF held-out RMSE (J/mol) | CuZr congruent (K) | "
        "Cu10Zr7 congruent (K) | invariant MAE fit / held-out (K) | gate reactions passing | gate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    s = row["rmse"]
    inv0 = out["start"]["invariants"]
    lines.append(
        f"| SGTE start | {s['fit/ACR']['start'] / 1e3:.2f} | {s['holdout/ACR']['start'] / 1e3:.2f} | "
        f"{s['holdout/ZPF_literature']['start']:.0f} | {inv0[key_reactions[0]]:.0f} | {inv0[key_reactions[1]]:.0f} | "
        f"{out['start']['invariant_errors']['fit_mae_K']:.0f} / {out['start']['invariant_errors']['holdout_mae_K']:.0f} | "
        f"{out['start']['gate_reactions_pass']}/13 | fail |"
    )

    def line(lab: str, r: dict) -> str:
        """One table row for a summarised run.

        Args:
            lab (str): Row label.
            r (dict): Summary record of the run.

        Returns:
            str: The Markdown table row.
        """
        q = r["rmse"]
        iv_ = r["invariants_opt"]
        k0, k1 = key_reactions
        t0 = f"{iv_[k0]:.0f}" if iv_[k0] else "absent"
        t1 = f"{iv_[k1]:.0f}" if iv_[k1] else "absent"
        mark = "PASS" if r["gate_pass"] else "fail"
        return (
            f"| {lab} | {q['fit/ACR']['optimum'] / 1e3:.2f} | {q['holdout/ACR']['optimum'] / 1e3:.2f} | "
            f"{q['holdout/ZPF_literature']['optimum']:.0f} | {t0} | {t1} | "
            f"{r['invariant_errors']['fit_mae_K']:.0f} / {r['invariant_errors']['holdout_mae_K']:.0f} | "
            f"{r['gate_reactions_pass']}/{r['gate_reactions']} | {mark} |"
        )

    lines.append(line("m6 x30 (v4)", m6))
    lines.append(line("m7 x30", row))
    at_bound = row["at_bound"] or "none"
    pass_word = "PASS" if out["pass_criterion"]["PASS"] else "FAIL"
    lines.append(
        f"\nm7 best start {row['best_start']}; L2(LIQUID) a = {row['theta_opt'][6]:.0f} J/mol; at bound: {at_bound}. "
        f"Pass criterion (activities better fit AND hold-out AND gate): {pass_word}."
    )
    failing = "; ".join(
        f"{k} {v['model_K']:.0f} K vs {v['nearest_printed_K']:.0f} K (u {v['u_K']:.0f})"
        if v["model_K"] is not None
        else f"{k} absent"
        for k, v in row["gate_detail"].items()
        if not v["pass"]
    )
    lines.append("Failing gate reactions (m7): " + failing)
    return lines


def main() -> None:
    """Analyse the m7 runs and write the summary JSON, the Markdown table and the REVIEW-ONLY TDB."""
    os.chdir(root)
    inv0 = analyze.inv_job(mst.start)
    _ok0, _, det0 = analyze.gate([inv0])
    run_files = sorted(Path(f"runs/{tag}").glob("*/result.json"), key=lambda p: p.as_posix())
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in run_files]
    best = max(runs, key=lambda r: r["lnL_opt"])
    th = np.array(best["theta_opt"])
    with Path(f"runs/{tag}/{best['name']}/trace.jsonl").open(encoding="utf-8") as fh:
        tr = [json.loads(line) for line in fh]
    bt = max((t for t in tr if t["lnL"] is not None), key=lambda t: t["lnL"])
    inv = analyze.inv_job(th)
    ok, _per, det = analyze.gate([inv])
    inv_all = {r["name"]: analyze.inv_job(r["theta_opt"]) for r in runs}
    metrics = analyze.metrics({"start": mst.start, "optimum": th})
    starts = {
        r["name"]: {
            "lnL": r["lnL_opt"],
            "theta": r["theta_opt"],
            "message": r["message"],
            "nit": r["nit"],
            "nfev": r["nfev"],
            "wall_s": r["wall_s"],
            "gate_reactions_pass": int(sum(v["pass"] for v in analyze.gate([inv_all[r["name"]]])[2][0].values())),
        }
        for r in runs
    }
    per_start = tr[0]["per"] if best["name"] == "SGTE" else _first_trace_line(f"runs/{tag}/SGTE/trace.jsonl")["per"]
    row = {
        "model": "m7",
        "zpf_factor": 30,
        "data": "data/v4_w30 (data/v4_m7 is a symlink)",
        "starts": starts,
        "best_start": best["name"],
        "symbols": mst.syms,
        "theta_opt": th.tolist(),
        "lnL_opt": best["lnL_opt"],
        "at_bound": [
            s
            for s, t, lo, hi, sc in zip(mst.syms, th, mst.lower, mst.upper, mst.scale, strict=True)
            if min(abs(t - lo), abs(t - hi)) < 1e-4 * sc
        ],
        "lnL_per_type_start": per_start,
        "lnL_per_type_opt": bt["per"],
        "rmse": {
            k: {"start": _rmse(metrics, k, "start"), "optimum": _rmse(metrics, k, "optimum"), "unit": metrics[k]["unit"]}
            for k in metrics
        },
        "invariants_opt": {k: (float(v) if v is not None else None) for k, v in inv.items()},
        "invariants_opt_all_starts": {
            n: {k: (float(v) if v is not None else None) for k, v in d.items()} for n, d in inv_all.items()
        },
        "invariant_errors": {"fit_mae_K": _mae(inv, "fit"), "holdout_mae_K": _mae(inv, "holdout")},
        "gate_pass": bool(ok[0]),
        "gate_reactions_pass": int(sum(v["pass"] for v in det[0].values())),
        "gate_reactions": len(det[0]),
        "gate_detail": det[0],
    }
    with Path("results/v4/summary_v4.json").open(encoding="utf-8") as fh:
        v4 = json.load(fh)
    m6 = [v for v in v4["variants"] if v["zpf_factor"] == 30][0]
    act_fit = row["rmse"]["fit/ACR"]["optimum"] < row["rmse"]["fit/ACR"]["start"]
    act_ho = row["rmse"]["holdout/ACR"]["optimum"] < row["rmse"]["holdout/ACR"]["start"]
    out = {
        "status": "POST-HOC (Amendment 4 fallback m7), model-adequacy test, not a tuned result; v3 is the pre-registered result",
        "selection_rule": "ZPF factor whose m6 optimum fails the fewest gate reactions (ties: larger activity improvement)",
        "selection": {f"x{v['zpf_factor']}": v["gate_reactions"] - v["gate_reactions_pass"] for v in v4["variants"]},
        "start": {
            "invariants": {k: (float(v) if v is not None else None) for k, v in inv0.items()},
            "gate_reactions_pass": int(sum(v["pass"] for v in det0[0].values())),
            "invariant_errors": {"fit_mae_K": _mae(inv0, "fit"), "holdout_mae_K": _mae(inv0, "holdout")},
        },
        "m7": row,
        "pass_criterion": {
            "activities_improved_fit": bool(act_fit),
            "activities_improved_holdout": bool(act_ho),
            "gate_passed": row["gate_pass"],
            "PASS": bool(act_fit and act_ho and row["gate_pass"]),
        },
        "printed": {
            k: sorted(r["T_K"] for r in analyze.PRINTED if r["reaction"] == k and r["split"] == "fit") for k in key_reactions
        },
    }
    Path(f"results/{tag}").mkdir(parents=True, exist_ok=True)
    with Path(f"results/{tag}/summary_v4_m7.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    mst.write(
        f"tdb/v4/CuZr-94Zen-LB_ORCHID-{tag}_REVIEW-ONLY.tdb",
        dict(zip(mst.syms, th, strict=True)),
        "v4 m7 (model-adequacy test, post-hoc) best optimum, ZPF x30. REVIEW ONLY: demonstration, not an assessment.",
    )
    lines = _table(row, m6, out)
    printed_k0, printed_k1 = out["printed"][key_reactions[0]], out["printed"][key_reactions[1]]
    Path(f"results/{tag}/summary_v4_m7.md").write_text(
        "\n".join(lines) + f"\n\nPrinted (fit): CuZr congruent {printed_k0} K; Cu10Zr7 congruent {printed_k1} K.\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
