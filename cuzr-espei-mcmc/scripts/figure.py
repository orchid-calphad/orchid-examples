"""The documented negative result of the classic ESPEI route for Cu-Zr, from cached/espei_demo_summary.json (no chains).

a  invariant reactions within the gate (of 13) for every variant of the orchid-espei-demo study: SGTE start, the
   pre-registered gradient fit (v3) and the post-hoc gradient variants (v4); the MCMC runs are annotated (v1: gate not
   yet defined; v2: 0 % of posterior draws pass, chains not converged). No variant passes (13/13).
b  v1 MCMC (ESPEI 0.9.1, emcee): temperature at which liquid and compound Gibbs energies are equal at the compound
   composition (an upper bound of its melting point), SGTE start vs posterior median, against the printed congruent
   melting temperatures in ORCHID (fit set). The posterior raises CuZr and CuZr2 by 150-180 K.
c  v2 MCMC: fraction of posterior draws that pass, per reaction.

  python scripts/figure.py [OUT_DIR]      # default figures/
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "common"))
import figstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAME = {"CU5ZR_C15B": "Cu$_5$Zr", "CU51ZR14": "Cu$_{51}$Zr$_{14}$", "CU8ZR3": "Cu$_8$Zr$_3$", "CU10ZR7": "Cu$_{10}$Zr$_7$",
        "CUZR_B2": "CuZr", "CUZR2": "CuZr$_2$"}
CONGRUENT = {"CU51ZR14": "L = Cu51Zr14 (congruent)", "CU10ZR7": "L = Cu10Zr7 (congruent)", "CUZR_B2": "L = CuZr (congruent)",
             "CUZR2": "L = CuZr2 (congruent)"}


def main(out):
    d = json.load(open(os.path.join(ROOT, "cached", "espei_demo_summary.json")))
    printed = json.load(open(os.path.join(ROOT, "inputs", "printed_invariants.json")))
    fig = plt.figure(figsize=(S.W2, 70 * S.MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.1])
    # a
    ax = fig.add_subplot(gs[0])
    rows = [v for v in d["variants"]]
    y = np.arange(len(rows))
    for i, v in enumerate(rows):
        if "gate_reactions_pass" in v:
            c = S.INK2 if v["key"] == "sgte_start" else S.ACCENT
            ax.barh(i, v["gate_reactions_pass"], color=c, height=0.6)
            ax.text(v["gate_reactions_pass"] + 0.2, i, f"{v['gate_reactions_pass']}/13", va="center", fontsize=5.5)
        else:
            txt = ("no gate yet; CuZr, CuZr$_2$\nmelt 150-180 K too high" if v["key"] == "v1_mcmc"
                   else f"0 % of posterior draws pass;\nsplit R-hat up to {v['split_rhat_range'][1]:.0f}")
            ax.text(0.2, i, txt, va="center", fontsize=5, color=S.C_MODEL)
    ax.axvline(13, color=S.INK, lw=0.5, ls="--"); ax.text(12.9, -0.9, "pass = 13", ha="right", fontsize=5)
    lab = {"sgte_start": "SGTE start", "v1_mcmc": "v1 MCMC, 8 par.", "v2_mcmc": "v2 MCMC, 16 par.", "v3_gradient": "v3 gradient (pre-reg.)"}
    ax.set_yticks(y); ax.set_yticklabels([lab.get(v["key"], v["key"].replace("v4_zpf_x", "v4 gradient, ZPF ×")) for v in rows], fontsize=5.5)
    ax.invert_yaxis(); ax.set_xlim(0, 14); ax.set_xlabel("Invariants within the gate (of 13)")
    S.panel_label(ax, "a", x=-0.55)
    # b
    ax = fig.add_subplot(gs[1])
    eg = d["v1_liquid_compound_equal_G_temperature_K"]
    comps = ["CU51ZR14", "CU10ZR7", "CUZR_B2", "CUZR2"]
    for i, c in enumerate(comps):
        pt = [r["T_K"] for r in printed if r["reaction"] == CONGRUENT[c] and r["split"] == "fit"]
        if pt:
            ax.plot([i, i], [min(pt), max(pt)] if len(pt) > 1 else [pt[0] - 3, pt[0] + 3], color=S.C_EXP, lw=4, alpha=0.5, solid_capstyle="butt")
        ax.plot(i - 0.12, eg["start_SGTE"][c], "o", color=S.INK2, ms=3.5)
        ax.plot(i + 0.12, eg["main_posterior_median"][c], "D", color=S.C_MODEL, ms=3.5)
    ax.plot([], [], color=S.C_EXP, lw=4, alpha=0.5, label="printed congruent melting (ORCHID)")
    ax.plot([], [], "o", color=S.INK2, ms=3.5, label="SGTE start")
    ax.plot([], [], "D", color=S.C_MODEL, ms=3.5, label="v1 MCMC posterior median")
    ax.set_xticks(range(len(comps))); ax.set_xticklabels([NAME[c] for c in comps], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Liquid = compound $G$ temperature (K)"); ax.set_xlim(-0.5, len(comps) - 0.5)
    ax.set_ylim(1140, 1500); ax.legend(fontsize=5, loc="upper left", handlelength=1.2)
    S.panel_label(ax, "b", x=-0.3)
    # c
    ax = fig.add_subplot(gs[2])
    pr = next(v for v in d["variants"] if v["key"] == "v2_mcmc")["per_reaction_pass_fraction"]
    ks = list(pr)
    ax.barh(range(len(ks)), [pr[k] for k in ks], color=S.C_MODEL, height=0.6)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=4.6); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("v2 MCMC: draws passing")
    S.panel_label(ax, "c", x=-0.9)
    S.save(fig, os.path.join(out, "fig_espei_negative"))
    print("wrote", os.path.join(out, "fig_espei_negative.{png,pdf}"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figures"))
