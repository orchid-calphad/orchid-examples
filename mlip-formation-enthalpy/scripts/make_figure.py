#!/usr/bin/env python3
"""Figure of the MLIP benchmark: formation enthalpies, robust sigma by stratum, and MLIP vs DFT.

Panel (a): MLIP / DFT formation enthalpies vs experiment on identical records; (b): robust sigma by stratum; (c): MLIP
vs DFT against MLIP vs experiment. Adapted from orchid-mlip-benchmark scripts/make_figure.py: it reads
results/summary.json (scripts/analyze.py) instead of CALIBRATION_MLIP.json; style from common/figstyle.py.

  python scripts/make_figure.py [OUT_DIR]     # reads OUT_DIR/summary.json (default results/), writes OUT_DIR or figures/
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import figstyle as fs
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
labels = {
    "grace-2l-omat": "GRACE-2L-OMAT",
    "mace-mpa-0": "MACE-MPA-0",
    "mace-omat-0": "MACE-OMAT-0",
    "sevennet-omni": "SevenNet-Omni",
    "orb-v3-mpa": "ORB-v3 (MPA)",
    "orb-v3-omat": "ORB-v3 (OMat)",
    "mattersim-5m": "MatterSim-v1 5M",
    "chgnet-0.3.0": "CHGNet 0.3.0",
    "uma-m-1p1": "UMA-M 1.1",
    "mp": "Materials Project",
    "oqmd": "OQMD",
}
strata = [
    ("all", "All"),
    ("chem:metal", "All-metal"),
    ("chem:magnetic_3d", "Magnetic 3d"),
    ("chem:pblock_anion", "p-block / anion"),
    ("match:on_hull_single", "On hull, single"),
    ("match:polymorph_ambiguous", "Polymorph-ambiguous"),
    ("match:above_hull", "Above hull"),
]

_cuzr_indirect = 9.7  # kJ/mol-atom: sigma_DFT (+) MLIP->DFT for GRACE on Cu-Zr (cuzr-phaseforgeplus, sigma_mlip.json)


def _draw_parity_panels(top: object, srcs: list[str], full: list[dict], exp: np.ndarray, same: dict) -> list[object]:
    """Panel (a): computed vs experimental formation enthalpy, one small axes per source.

    Args:
        top (object): Matplotlib subfigure that holds the grid.
        srcs (list[str]): Sources to plot (DFT references first, then the MLIPs).
        full (list[dict]): Records of pairs_identical_set.csv, sorted by record id.
        exp (np.ndarray): Experimental formation enthalpies of the records, kJ/mol-atom.
        same (dict): Identical-set statistics per source, from summary.json.

    Returns:
        The axes, in the order of srcs.
    """
    ncol = 5
    nrow_p = math.ceil(len(srcs) / ncol)
    gs = top.add_gridspec(nrow_p, ncol)
    lo, hi = -110, 15
    axes = []
    for k, s in enumerate(srcs):
        ax = top.add_subplot(gs[k // ncol, k % ncol])
        y = np.array([float(r[s]) for r in full])
        col = fs.src.get(s, fs.ink2)
        ax.plot([lo, hi], [lo, hi], color=fs.grey_m, lw=0.5, zorder=0)
        ax.fill_between([lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10], color=fs.grey_l, lw=0, zorder=0)
        ax.scatter(exp, y, s=2.2, color=col, alpha=0.55, lw=0, rasterized=True)
        st = same[s]["all"]
        ax.text(
            0.04,
            0.97,
            f"{labels.get(s, s)}\n$\\sigma$ = {st['sigma_robust']:.1f}, MAE = {st['mae']:.1f}\nbias = {st['bias_median']:+.1f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=4.8,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xticks([-100, -50, 0])
        ax.set_yticks([-100, -50, 0])
        if k % ncol == 0:
            ax.set_ylabel("Computed $\\Delta H_f$\n(" + fs.unit_kja + ")")
        if k + ncol >= len(srcs) and k % ncol == ncol // 2:
            ax.set_xlabel("Experimental $\\Delta H_f$ (" + fs.unit_kja + ")")
        axes.append(ax)
    fs.panel_label(axes[0], "a", x=-0.3)
    return axes


def _draw_stratum_panel(ax: object, srcs: list[str], same: dict) -> None:
    """Panel (b): robust sigma of every source in every stratum, with cluster-bootstrap intervals.

    Args:
        ax (object): Axes to draw on.
        srcs (list[str]): Sources to plot.
        same (dict): Identical-set statistics per source, from summary.json.
    """
    xs = np.arange(len(strata))
    w = 0.8 / len(srcs)
    for j, s in enumerate(srcs):
        col = fs.src.get(s, None)
        shade = col or plt.cm.Greys(0.35 + 0.6 * j / max(len(srcs) - 1, 1))
        for i, (key, _) in enumerate(strata):
            st = same[s]["all"] if key == "all" else same[s]["strata"].get(key)
            if not st or st["n"] < 10:
                continue
            x = xs[i] - 0.4 + w * (j + 0.5)
            ci = (st.get("ci95_cluster_system") or st["ci95"])["sigma_robust"]
            ax.plot([x, x], ci, color=shade, lw=0.6)
            ax.plot(x, st["sigma_robust"], "o", ms=2.2, color=shade, mec="none", label=labels.get(s, s) if i == 0 else None)
    ax.axhline(_cuzr_indirect, color=fs.accent, lw=0.6, ls="--", label="Cu–Zr indirect estimate")
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in strata], rotation=25, ha="right")
    ax.set_ylabel("Robust $\\sigma$ vs experiment\n(" + fs.unit_kja + ")")
    ax.set_ylim(0, None)
    ax.legend(
        ncol=4,
        fontsize=4.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        handletextpad=0.1,
        columnspacing=0.6,
        borderaxespad=0.2,
    )
    fs.panel_label(ax, "b")


def _draw_mlip_vs_dft_panel(ax: object, cal: dict, same: dict, models: list[str]) -> None:
    """Panel (c): sigma of each MLIP against MP and OQMD (as DFT) and against experiment.

    Args:
        ax (object): Axes to draw on.
        cal (dict): Content of summary.json (needs mlip_vs_dft).
        same (dict): Identical-set statistics per source, from summary.json.
        models (list[str]): MLIP models, in plotting order.
    """
    vd = cal["mlip_vs_dft"]
    yy = np.arange(len(models))
    for i, m in enumerate(models):
        for s, off in (("mp", -0.15), ("oqmd", 0.15)):
            st = vd[m][s]
            ax.plot((st.get("ci95_cluster_system") or st["ci95"])["sigma_robust"], [i + off] * 2, color=fs.src[s], lw=0.6)
            ax.plot(
                st["sigma_robust"],
                i + off,
                "o",
                ms=2.2,
                color=fs.src[s],
                mec="none",
                label=("MLIP − " + fs.src_label[s]) if i == 0 else None,
            )
        ax.plot(same[m]["all"]["ci95_cluster_system"]["sigma_robust"], [i] * 2, color=fs.ink, lw=0.6)
        ax.plot(
            same[m]["all"]["sigma_robust"],
            i,
            "D",
            ms=2.2,
            color=fs.ink,
            mec="none",
            label="MLIP − experiment" if i == 0 else None,
        )
    ax.axvline(same["mp"]["all"]["sigma_robust"], color=fs.c_mp, lw=0.5, ls=":")
    ax.axvline(same["oqmd"]["all"]["sigma_robust"], color=fs.c_oqmd, lw=0.5, ls=":")
    ax.set_yticks(yy)
    ax.set_yticklabels([labels[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Robust $\\sigma$ (" + fs.unit_kja + ")")
    ax.set_xlim(0, None)
    ax.legend(
        fontsize=4.8,
        loc="lower center",
        bbox_to_anchor=(0.55, 1.0),
        ncol=3,
        handletextpad=0.1,
        columnspacing=0.6,
        borderaxespad=0.2,
    )
    fs.panel_label(ax, "c")


def main() -> None:
    """Read summary.json, draw the three-panel figure and write it as PNG and PDF."""
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None
    res_dir = Path(out_dir) if out_dir else root / "results"
    with (res_dir / "summary.json").open() as f:
        cal = json.load(f)
    same = cal["same_set"]
    models = sorted([m for m in same if m not in ("mp", "oqmd")], key=lambda m: same[m]["all"]["sigma_robust"])
    srcs = ["mp", "oqmd", *models]
    with (root / "inputs" / "pairs_identical_set.csv").open() as f:
        full = sorted(csv.DictReader(f), key=lambda r: r["rid"])
    exp = np.array([float(r["exp"]) for r in full])

    fig = plt.figure(
        figsize=(fs.w2, min(fs.hmax, (38 * math.ceil(len(srcs) / 5) + 62) * fs.mm)),
        constrained_layout=True,
    )
    top, bot = fig.subfigures(2, 1, height_ratios=[math.ceil(len(srcs) / 5), 1.45])
    gsb = bot.add_gridspec(1, 5)
    _draw_parity_panels(top, srcs, full, exp, same)
    _draw_stratum_panel(bot.add_subplot(gsb[0, :3]), srcs, same)
    _draw_mlip_vs_dft_panel(bot.add_subplot(gsb[0, 3:]), cal, same, models)
    stem = str((Path(out_dir) if out_dir else root / "figures") / "mlip_benchmark")
    fs.save(fig, stem)
    print("n records", len(full), "->", stem + ".{png,pdf}")


if __name__ == "__main__":
    main()
