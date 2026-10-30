"""Cu-Zr one-shot figure: phase map with evidence points, and per-phase recall.

(a) Cu-Zr phase diagram of the GRACE-2L-OMAT MLIP-only PhaseForge TDB (pycalphad, same settings as
score.py) with the ORCHID evidence points coloured by whether the model recovers their reference phases;
(b) per-phase recall (field + boundary points pooled), GRACE vs SGTE (Zeng 1994).
Usage: python figure.py [--grace results/cuzr_grace.json] [--sgte results/cuzr_sgte_94Zen.json] [--out figures]
Outputs <out>/fig_cuzr_oneshot.{pdf,png}; the phase map grid is cached in results/map_cuzr_grace.npz (strings, no pickle;
computed with pycalphad only when missing). Adapted from orchid-bench-pd scripts/figure.py: font from common/figstyle.py
(deterministic output), output directory option.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).absolute().parent))
sys.path.insert(0, str(Path(__file__).absolute().parents[2] / "common"))
import figstyle  # noqa: E402
import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import phase_identity as pid  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from release import example_root  # noqa: E402
from score import Model  # noqa: E402

temp_min, temp_max, temp_step = 700.0, 1800.0, 4.0  # K, K, K: grid of the phase map
n_x = 251  # composition nodes of the phase map
short_names = {
    "LIQUID": "L",
    "FCC_A1": "(Cu)",
    "BCC_A2": "(βZr)",
    "HCP_A3": "(αZr)",
    "C15B_CU5ZR": "Cu$_5$Zr",
    "CU51ZR14": "Cu$_{51}$Zr$_{14}$",
    "CU8ZR3": "Cu$_8$Zr$_3$",
    "CU10ZR7": "Cu$_{10}$Zr$_7$",
    "CUZR_B2": "CuZr (B2)",
    "CUZR2": "CuZr$_2$",
    "CU24ZR13": "Cu$_{24}$Zr$_{13}$",
    "CU2ZR": "Cu$_2$Zr",
    "CU5ZR8": "Cu$_5$Zr$_8$",
    "CU5ZR7": "Cu$_5$Zr$_7$",
}
c_ok, c_part, c_miss = "#2a78d6", "#eda100", "#d03b3b"
c_grace, c_sgte = "#2a78d6", "#8a8a85"
ink_text, muted_text = "#1f1f1e", "#6b6b66"


def phase_map(tdb: str, names_as: str, cache: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase map of a TDB on the temperature and composition grid, from the cache when it exists.

    Args:
        tdb (str): TDB file, used only when the cache is missing.
        names_as (str): Registry TDB whose phase names the TDB uses.
        cache (str | Path): Cache file (.npz) with the arrays T, x and lab.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Temperatures in K, mole fractions of Zr, and the assemblage label
        per node ("+"-joined benchmark classes).
    """
    if Path(cache).exists():
        with np.load(cache, allow_pickle=False) as z:
            return z["T"], z["x"], z["lab"].astype(object)
    m = Model(tdb, names_as)
    temps = np.arange(temp_min, temp_max + 0.1, temp_step)
    xs = np.linspace(0, 1, n_x)
    lab = np.empty((len(temps), len(xs)), dtype=object)
    for i in range(0, len(temps), 20):
        a = m.equilibrium(temps[i : i + 20], xs)
        for ii in range(a.shape[0]):
            for j in range(a.shape[1]):
                c = a[ii, j]
                if not c:
                    c = m.equilibrium_drop_one(temps[i + ii], xs[j])
                lab[i + ii, j] = "+".join(sorted({pid.bench_class(m.reg_id(n, d)) for n, d in c}))
    np.savez_compressed(cache, T=temps, x=xs, lab=lab.astype(str))
    return temps, xs, lab


def _draw_phase_map(ax: plt.Axes, temps: np.ndarray, xs: np.ndarray, lab: np.ndarray) -> None:
    """Draw the phase map: one-phase fields tinted, boundaries as contours, field names at their widest cell.

    Args:
        ax (plt.Axes): Axes to draw on.
        temps (np.ndarray): Temperatures in K.
        xs (np.ndarray): Mole fractions of Zr.
        lab (np.ndarray): Assemblage label per node.
    """
    keys = sorted(set(lab.ravel()))
    idx = np.vectorize(keys.index)(lab)
    ax.pcolormesh(xs, temps, np.zeros_like(idx, float), cmap="Greys", vmin=0, vmax=1, shading="auto")
    for k in keys:
        ind = (lab == k).astype(float)
        ax.contour(xs, temps, ind, levels=[0.5], colors=[muted_text], linewidths=0.45)
    for k in keys:  # light tint on one-phase fields (liquid, solid solutions)
        if "+" not in k:
            ax.contourf(xs, temps, (lab == k).astype(float), levels=[0.5, 1.5], colors=["#e8eef7"])
    for k in keys:  # field labels at the cell of maximum clearance
        ind = lab == k
        if ind.sum() < 60:
            continue
        dmap = distance_transform_edt(np.pad(ind, 1))[1:-1, 1:-1]
        i, j = np.unravel_index(np.argmax(dmap), dmap.shape)
        if dmap[i, j] < 4:
            continue
        txt = "+".join(short_names.get(p, p) for p in k.split("+"))
        ax.text(xs[j], temps[i], txt, ha="center", va="center", fontsize=4.6, color=muted_text)


def _draw_evidence(ax: plt.Axes, fig: plt.Figure, result: dict) -> None:
    """Draw the evidence points coloured by recall, with the axis labels, legend and recall summary.

    Args:
        ax (plt.Axes): Phase-map axes.
        fig (plt.Figure): Figure, for the legend.
        result (dict): Scored GRACE result.
    """
    pts = result["points"]
    for kind, mk, ms in (("boundary", "D", 9), ("field", "o", 22)):
        for cat, col, face in (("ok", c_ok, True), ("partial", c_part, True), ("miss", c_miss, False)):
            sel = [
                p
                for p in pts
                if p["kind"] == kind
                and ({"ok": p["recall_point"] == 1, "partial": 0 < p["recall_point"] < 1, "miss": p["recall_point"] == 0}[cat])
            ]
            if not sel:
                continue
            ax.scatter(
                [p["x_Zr"] for p in sel],
                [p["T_K"] for p in sel],
                s=ms,
                marker=mk if face else ("X" if kind == "field" else "x"),
                facecolors=col,
                edgecolors="white" if face else col,
                linewidths=0.4 if face else 0.7,
                zorder=3,
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(temp_min, temp_max)
    ax.set_xlabel("x$_{Zr}$ (mole fraction)")
    ax.set_ylabel("T (K)")
    gf = result["metrics"]["physical"]["field"]["all"]
    gb = result["metrics"]["physical"]["boundary"]["all"]
    ax.set_title("a  GRACE-2L-OMAT, MLIP-only PhaseForge TDB (no fitting)", loc="left", fontsize=7, fontweight="bold")
    leg = [
        Line2D([], [], marker="o", ls="", mfc=c_ok, mec="white", ms=4.5, label="field point, phases recovered"),
        Line2D([], [], marker="o", ls="", mfc=c_part, mec="white", ms=4.5, label="field point, partly"),
        Line2D([], [], marker="X", ls="", mfc=c_miss, mec=c_miss, ms=4.5, label="field point, none"),
        Line2D([], [], marker="D", ls="", mfc=c_ok, mec="white", ms=3, label="boundary point (±10 K, ±0.01), recovered"),
        Line2D([], [], marker="D", ls="", mfc=c_part, mec="white", ms=3, label="boundary point, partly"),
        Line2D([], [], marker="x", ls="", mec=c_miss, ms=3.5, label="boundary point, none"),
    ]
    fig.legend(
        handles=leg,
        loc="lower left",
        bbox_to_anchor=(0.575, 0.0),
        fontsize=4.8,
        frameon=False,
        ncol=2,
        title="evidence points (colour: reference phases the model recovers)",
        title_fontsize=4.8,
        columnspacing=1.0,
    )
    ax.text(
        0.99,
        0.015,
        f"prevalence-weighted recall: field {gf['weighted']['recall']:.2f} "
        f"[{gf['ci95']['recall'][0]:.2f}, {gf['ci95']['recall'][1]:.2f}], n = {gf['n_points']}; "
        f"boundary {gb['weighted']['recall']:.2f} [{gb['ci95']['recall'][0]:.2f}, {gb['ci95']['recall'][1]:.2f}], n = {gb['n_points']}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.8,
        color=ink_text,
        bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.2},
    )


def _pooled(result: dict) -> dict:
    """Recall and support per phase, with field and boundary points pooled.

    Args:
        result (dict): Scored result.

    Returns:
        dict: Phase -> (recall, support) for the phases with support.
    """
    out = {}
    for kind in ("field", "boundary"):
        for ph, v in result["metrics"]["physical"][kind]["all"]["per_phase"].items():
            o = out.setdefault(ph, [0, 0])
            o[0] += v["tp"]
            o[1] += v["support"]
    return {k: (v[0] / v[1] if v[1] else np.nan, v[1]) for k, v in out.items() if v[1]}


def _draw_recall(bx: plt.Axes, grace: dict, sgte: dict) -> None:
    """Draw the per-phase recall bars of GRACE and SGTE.

    Args:
        bx (plt.Axes): Bar-chart axes.
        grace (dict): Scored GRACE result.
        sgte (dict): Scored SGTE result.
    """
    pg, ps = _pooled(grace), _pooled(sgte)
    phases = sorted(pg, key=lambda k: -pg[k][1])
    y = np.arange(len(phases))
    in_model = set(grace["model_classes"]["physical"])
    bx.barh(y - 0.2, [pg[p][0] for p in phases], height=0.38, color=c_grace, label="GRACE MLIP-only")
    bx.barh(y + 0.2, [ps.get(p, (0, 0))[0] for p in phases], height=0.38, color=c_sgte, label="SGTE (Zeng 1994)")
    bx.set_yticks(y)
    bx.set_yticklabels([f"{short_names.get(p, p)}{'' if p in in_model else ' *'}  ({pg[p][1]})" for p in phases], fontsize=5.2)
    bx.invert_yaxis()
    bx.set_xlim(0, 1.0)
    bx.set_xlabel("recall (field + boundary points)")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.xaxis.grid(True, lw=0.3, color="#d9d9d4")
    bx.set_axisbelow(True)
    bx.set_title("b  per-phase recall", loc="left", fontsize=7, fontweight="bold")
    bx.legend(loc="center right", bbox_to_anchor=(1.0, 0.22), fontsize=5, frameon=False)
    bx.text(
        0.0,
        -0.29,
        "(n) reference support; * not in either model's phase list",
        transform=bx.transAxes,
        fontsize=4.6,
        color=muted_text,
    )


def main() -> None:
    """Draw the two-panel figure and write it as PNG and PDF."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--grace", default=str(example_root / "results" / "cuzr_grace.json"))
    ap.add_argument("--sgte", default=str(example_root / "results" / "cuzr_sgte_94Zen.json"))
    ap.add_argument("--out", default=str(example_root / "figures"))
    a = ap.parse_args()
    grace = json.loads(Path(a.grace).read_text())
    sgte = json.loads(Path(a.sgte).read_text())
    temps, xs, lab = phase_map(
        str(example_root / grace["tdb"]), grace["names_as"], example_root / "results" / "map_cuzr_grace.npz"
    )
    plt.rcParams.update(
        {
            "font.family": figstyle.font,
            "font.size": 6.5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": ink_text,
            "text.color": ink_text,
            "axes.labelcolor": ink_text,
            "xtick.color": ink_text,
            "ytick.color": ink_text,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(183 / 25.4, 78 / 25.4))
    ax = fig.add_axes([0.075, 0.13, 0.485, 0.80])
    bx = fig.add_axes([0.70, 0.36, 0.29, 0.57])
    _draw_phase_map(ax, temps, xs, lab)
    _draw_evidence(ax, fig, grace)
    _draw_recall(bx, grace, sgte)
    figstyle.save(fig, Path(a.out) / "fig_cuzr_oneshot")
    # os.path.join keeps the separators of --out as given, so the printed text is unchanged
    print("wrote", os.path.join(a.out, "fig_cuzr_oneshot.{png,pdf}"))  # noqa: PTH118


if __name__ == "__main__":
    main()
