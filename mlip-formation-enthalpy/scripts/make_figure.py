#!/usr/bin/env python3
"""Figure: MLIP / DFT formation enthalpies vs experiment on identical records (a), robust sigma by stratum (b), and
MLIP vs DFT against MLIP vs experiment (c). Adapted from orchid-mlip-benchmark scripts/make_figure.py: it reads
results/summary.json (scripts/analyze.py) instead of CALIBRATION_MLIP.json; style from common/figstyle.py.

  python scripts/make_figure.py [OUT_DIR]     # reads OUT_DIR/summary.json (default results/), writes OUT_DIR or figures/
"""
import csv, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "common"))
import figstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL = {"grace-2l-omat": "GRACE-2L-OMAT", "mace-mpa-0": "MACE-MPA-0", "mace-omat-0": "MACE-OMAT-0",
         "sevennet-omni": "SevenNet-Omni", "orb-v3-mpa": "ORB-v3 (MPA)", "orb-v3-omat": "ORB-v3 (OMat)",
         "mattersim-5m": "MatterSim-v1 5M", "chgnet-0.3.0": "CHGNet 0.3.0", "uma-m-1p1": "UMA-M 1.1",
         "mp": "Materials Project", "oqmd": "OQMD"}
STRATA = [("all", "All"), ("chem:metal", "All-metal"), ("chem:magnetic_3d", "Magnetic 3d"),
          ("chem:pblock_anion", "p-block / anion"), ("match:on_hull_single", "On hull, single"),
          ("match:polymorph_ambiguous", "Polymorph-ambiguous"), ("match:above_hull", "Above hull")]


CUZR_INDIRECT = 9.7   # kJ/mol-atom: sigma_DFT (+) MLIP->DFT for GRACE on Cu-Zr (cuzr-phaseforgeplus, sigma_mlip.json)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None
    res_dir = out_dir or os.path.join(ROOT, "results")
    cal = json.load(open(os.path.join(res_dir, "summary.json")))
    same = cal["same_set"]
    models = sorted([m for m in same if m not in ("mp", "oqmd")], key=lambda m: same[m]["all"]["sigma_robust"])
    srcs = ["mp", "oqmd"] + models
    full = sorted(csv.DictReader(open(os.path.join(ROOT, "inputs", "pairs_identical_set.csv"))), key=lambda r: r["rid"])
    exp = np.array([float(r["exp"]) for r in full])

    ncol = 5
    nrow_p = math.ceil(len(srcs) / ncol)
    fig = plt.figure(figsize=(S.W2, min(S.HMAX, (38 * nrow_p + 62) * S.MM)), constrained_layout=True)
    top, bot = fig.subfigures(2, 1, height_ratios=[nrow_p, 1.45])
    gs = top.add_gridspec(nrow_p, ncol)
    gsb = bot.add_gridspec(1, 5)
    lo, hi = -110, 15
    axes = []
    for k, s in enumerate(srcs):
        ax = top.add_subplot(gs[k // ncol, k % ncol])
        y = np.array([float(r[s]) for r in full])
        col = S.SRC.get(s, S.INK2)
        ax.plot([lo, hi], [lo, hi], color=S.GREY_M, lw=0.5, zorder=0)
        ax.fill_between([lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10], color=S.GREY_L, lw=0, zorder=0)
        ax.scatter(exp, y, s=2.2, color=col, alpha=0.55, lw=0, rasterized=True)
        st = same[s]["all"]
        ax.text(0.04, 0.97, "%s\n$\\sigma$ = %.1f, MAE = %.1f\nbias = %+.1f" % (LABEL.get(s, s), st["sigma_robust"],
                                                                             st["mae"], st["bias_median"]),
                transform=ax.transAxes, va="top", ha="left", fontsize=4.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([-100, -50, 0]); ax.set_yticks([-100, -50, 0])
        if k % ncol == 0:
            ax.set_ylabel("Computed $\\Delta H_f$\n(" + S.UNIT_KJA + ")")
        if k + ncol >= len(srcs) and k % ncol == ncol // 2:
            ax.set_xlabel("Experimental $\\Delta H_f$ (" + S.UNIT_KJA + ")")
        axes.append(ax)
    S.panel_label(axes[0], "a", x=-0.3)

    ax = bot.add_subplot(gsb[0, :3])
    xs = np.arange(len(STRATA))
    w = 0.8 / len(srcs)
    for j, s in enumerate(srcs):
        col = S.SRC.get(s, None)
        shade = col or plt.cm.Greys(0.35 + 0.6 * j / max(len(srcs) - 1, 1))
        for i, (key, _) in enumerate(STRATA):
            st = same[s]["all"] if key == "all" else same[s]["strata"].get(key)
            if not st or st["n"] < 10:
                continue
            x = xs[i] - 0.4 + w * (j + 0.5)
            ci = (st.get("ci95_cluster_system") or st["ci95"])["sigma_robust"]
            ax.plot([x, x], ci, color=shade, lw=0.6)
            ax.plot(x, st["sigma_robust"], "o", ms=2.2, color=shade, mec="none",
                    label=LABEL.get(s, s) if i == 0 else None)
    ax.axhline(CUZR_INDIRECT, color=S.ACCENT, lw=0.6, ls="--", label="Cu–Zr indirect estimate")
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in STRATA], rotation=25, ha="right")
    ax.set_ylabel("Robust $\\sigma$ vs experiment\n(" + S.UNIT_KJA + ")")
    ax.set_ylim(0, None)
    ax.legend(ncol=4, fontsize=4.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.1,
              columnspacing=0.6, borderaxespad=0.2)
    S.panel_label(ax, "b")

    ax = bot.add_subplot(gsb[0, 3:])
    vd = cal["mlip_vs_dft"]
    yy = np.arange(len(models))
    for i, m in enumerate(models):
        for s, off in (("mp", -0.15), ("oqmd", 0.15)):
            st = vd[m][s]
            ax.plot((st.get("ci95_cluster_system") or st["ci95"])["sigma_robust"], [i + off] * 2, color=S.SRC[s], lw=0.6)
            ax.plot(st["sigma_robust"], i + off, "o", ms=2.2, color=S.SRC[s], mec="none",
                    label=("MLIP − " + S.SRCL[s]) if i == 0 else None)
        ax.plot(same[m]["all"]["ci95_cluster_system"]["sigma_robust"], [i] * 2, color=S.INK, lw=0.6)
        ax.plot(same[m]["all"]["sigma_robust"], i, "D", ms=2.2, color=S.INK, mec="none",
                label="MLIP − experiment" if i == 0 else None)
    ax.axvline(same["mp"]["all"]["sigma_robust"], color=S.C_MP, lw=0.5, ls=":")
    ax.axvline(same["oqmd"]["all"]["sigma_robust"], color=S.C_OQMD, lw=0.5, ls=":")
    ax.set_yticks(yy); ax.set_yticklabels([LABEL[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Robust $\\sigma$ (" + S.UNIT_KJA + ")")
    ax.set_xlim(0, None)
    ax.legend(fontsize=4.8, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=3, handletextpad=0.1,
              columnspacing=0.6, borderaxespad=0.2)
    S.panel_label(ax, "c")
    stem = os.path.join(out_dir or os.path.join(ROOT, "figures"), "mlip_benchmark")
    S.save(fig, stem)
    print("n records", len(full), "->", stem + ".{png,pdf}")


if __name__ == "__main__":
    main()
