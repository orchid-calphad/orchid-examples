"""The documented negative result of the classic ESPEI route for Cu-Zr, from cached/espei_demo_summary.json (no chains).

a  invariant reactions within the gate (of 13) for every variant of the orchid-espei-demo study: SGTE start, the
   pre-registered gradient fit (v3), the post-hoc gradient variants (v4) and the model-adequacy fallback m7; the MCMC
   runs are annotated (v1: gate not yet defined; v2: 0 % of posterior draws pass, chains not converged). No variant
   passes (13/13).
b  v1 MCMC (ESPEI 0.9.1, emcee): temperature at which liquid and compound Gibbs energies are equal at the compound
   composition (an upper bound of its melting point), SGTE start vs posterior median, against the printed congruent
   melting temperatures in ORCHID (fit set). The posterior raises CuZr and CuZr2 by 150-180 K.
c  v2 MCMC: fraction of posterior draws that pass, per reaction.

  python scripts/figure.py [OUT_DIR]      # default figures/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

root = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(root.parent / "common"))
import figstyle as style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

phase_names = {
    "CU5ZR_C15B": "Cu$_5$Zr",
    "CU51ZR14": "Cu$_{51}$Zr$_{14}$",
    "CU8ZR3": "Cu$_8$Zr$_3$",
    "CU10ZR7": "Cu$_{10}$Zr$_7$",
    "CUZR_B2": "CuZr",
    "CUZR2": "CuZr$_2$",
}
congruent_reaction = {
    "CU51ZR14": "L = Cu51Zr14 (congruent)",
    "CU10ZR7": "L = Cu10Zr7 (congruent)",
    "CUZR_B2": "L = CuZr (congruent)",
    "CUZR2": "L = CuZr2 (congruent)",
}
fig_height = 70 * style.mm  # figure height in inches (70 mm)


def _panel_gate(fig: plt.Figure, spec: object, d: dict) -> None:
    """Panel a: invariant reactions within the gate for each variant of the study.

    Args:
        fig (plt.Figure): Figure to draw on.
        spec (object): Grid cell for this panel.
        d (dict): Parsed cached/espei_demo_summary.json.
    """
    ax = fig.add_subplot(spec)
    rows = list(d["variants"])
    y = np.arange(len(rows))
    for i, v in enumerate(rows):
        if "gate_reactions_pass" in v:
            c = style.ink2 if v["key"] == "sgte_start" else style.accent
            ax.barh(i, v["gate_reactions_pass"], color=c, height=0.6)
            ax.text(v["gate_reactions_pass"] + 0.2, i, f"{v['gate_reactions_pass']}/13", va="center", fontsize=5.5)
        else:
            txt = (
                "no gate yet; CuZr, CuZr$_2$\nmelt 150-180 K too high"
                if v["key"] == "v1_mcmc"
                else f"0 % of posterior draws pass;\nsplit R-hat up to {v['split_rhat_range'][1]:.0f}"
            )
            ax.text(0.2, i, txt, va="center", fontsize=5, color=style.c_model)
    ax.axvline(13, color=style.ink, lw=0.5, ls="--")
    ax.text(12.9, -0.9, "pass = 13", ha="right", fontsize=5)
    lab = {
        "sgte_start": "SGTE start",
        "v1_mcmc": "v1 MCMC, 8 par.",
        "v2_mcmc": "v2 MCMC, 16 par.",
        "v3_gradient": "v3 gradient (pre-reg.)",
        "v4_m7_zpf_x30": "m7 (m6 + 2L), ZPF ×30",
    }
    ax.set_yticks(y)
    ax.set_yticklabels(
        [lab.get(v["key"], v["key"].replace("v4_zpf_x", "v4 gradient, ZPF ×")) for v in rows],
        fontsize=5.5,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 14)
    ax.set_xlabel("Invariants within the gate (of 13)")
    style.panel_label(ax, "a", x=-0.55)


def _panel_melting(fig: plt.Figure, spec: object, d: dict, printed: list[dict]) -> None:
    """Panel b: liquid = compound Gibbs-energy temperatures, SGTE start vs posterior median, vs printed melting.

    Args:
        fig (plt.Figure): Figure to draw on.
        spec (object): Grid cell for this panel.
        d (dict): Parsed cached/espei_demo_summary.json.
        printed (list[dict]): Printed invariant records (reaction, T_K, split).
    """
    ax = fig.add_subplot(spec)
    eg = d["v1_liquid_compound_equal_G_temperature_K"]
    comps = ["CU51ZR14", "CU10ZR7", "CUZR_B2", "CUZR2"]
    for i, c in enumerate(comps):
        pt = [r["T_K"] for r in printed if r["reaction"] == congruent_reaction[c] and r["split"] == "fit"]
        if pt:
            ax.plot(
                [i, i],
                [min(pt), max(pt)] if len(pt) > 1 else [pt[0] - 3, pt[0] + 3],
                color=style.c_exp,
                lw=4,
                alpha=0.5,
                solid_capstyle="butt",
            )
        ax.plot(i - 0.12, eg["start_SGTE"][c], "o", color=style.ink2, ms=3.5)
        ax.plot(i + 0.12, eg["main_posterior_median"][c], "D", color=style.c_model, ms=3.5)
    ax.plot([], [], color=style.c_exp, lw=4, alpha=0.5, label="printed congruent melting (ORCHID)")
    ax.plot([], [], "o", color=style.ink2, ms=3.5, label="SGTE start")
    ax.plot([], [], "D", color=style.c_model, ms=3.5, label="v1 MCMC posterior median")
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([phase_names[c] for c in comps], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Liquid = compound $G$ temperature (K)")
    ax.set_xlim(-0.5, len(comps) - 0.5)
    ax.set_ylim(1140, 1500)
    ax.legend(fontsize=5, loc="upper left", handlelength=1.2)
    style.panel_label(ax, "b", x=-0.3)


def _panel_draws(fig: plt.Figure, spec: object, d: dict) -> None:
    """Panel c: fraction of v2 posterior draws that pass each reaction.

    Args:
        fig (plt.Figure): Figure to draw on.
        spec (object): Grid cell for this panel.
        d (dict): Parsed cached/espei_demo_summary.json.
    """
    ax = fig.add_subplot(spec)
    pr = next(v for v in d["variants"] if v["key"] == "v2_mcmc")["per_reaction_pass_fraction"]
    ks = list(pr)
    ax.barh(range(len(ks)), [pr[k] for k in ks], color=style.c_model, height=0.6)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=4.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("v2 MCMC: draws passing")
    style.panel_label(ax, "c", x=-0.9)


def main(out: str) -> None:
    """Draw the three-panel negative-result figure and save it as PNG and PDF.

    Args:
        out (str): Output directory; the stem fig_espei_negative is written inside it.
    """
    d = json.loads((root / "cached" / "espei_demo_summary.json").read_text(encoding="utf-8"))
    printed = json.loads((root / "inputs" / "printed_invariants.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(style.w2, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.1])
    _panel_gate(fig, gs[0], d)
    _panel_melting(fig, gs[1], d, printed)
    _panel_draws(fig, gs[2], d)
    # os.path.join keeps the printed path text exactly as passed on the command line (pathlib would normalise it).
    style.save(fig, os.path.join(out, "fig_espei_negative"))  # noqa: PTH118
    print("wrote", os.path.join(out, "fig_espei_negative.{png,pdf}"))  # noqa: PTH118


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(root / "figures"))
