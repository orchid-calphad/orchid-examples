"""Figure: (a) Cu-Zr phase diagram of the GRACE-2L-OMAT MLIP-only PhaseForge TDB (pycalphad, same settings as
score.py) with the ORCHID evidence points coloured by whether the model recovers their reference phases;
(b) per-phase recall (field + boundary points pooled), GRACE vs SGTE (Zeng 1994).
Usage: python figure.py [--grace results/cuzr_grace.json] [--sgte results/cuzr_sgte_94Zen.json] [--out figures]
Outputs <out>/fig_cuzr_oneshot.{pdf,png}; the phase map grid is cached in results/map_cuzr_grace.npz (strings, no pickle;
computed with pycalphad only when missing). Adapted from orchid-bench-pd scripts/figure.py: font from common/figstyle.py
(deterministic output), output directory option."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))
import figstyle  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import score as sc
import phase_identity as pid

ROOT = sc.ROOT
T0, T1, DT = 700.0, 1800.0, 4.0
NX = 251
SHORT = {"LIQUID": "L", "FCC_A1": "(Cu)", "BCC_A2": "(βZr)", "HCP_A3": "(αZr)", "C15B_CU5ZR": "Cu$_5$Zr",
         "CU51ZR14": "Cu$_{51}$Zr$_{14}$", "CU8ZR3": "Cu$_8$Zr$_3$", "CU10ZR7": "Cu$_{10}$Zr$_7$", "CUZR_B2": "CuZr (B2)",
         "CUZR2": "CuZr$_2$", "CU24ZR13": "Cu$_{24}$Zr$_{13}$", "CU2ZR": "Cu$_2$Zr", "CU5ZR8": "Cu$_5$Zr$_8$", "CU5ZR7": "Cu$_5$Zr$_7$"}
C_OK, C_PART, C_MISS = "#2a78d6", "#eda100", "#d03b3b"
C_GRACE, C_SGTE = "#2a78d6", "#8a8a85"
INK, MUTED = "#1f1f1e", "#6b6b66"


def phase_map(tdb, names_as, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["T"], z["x"], z["lab"].astype(object)
    m = sc.Model(tdb, names_as)
    Ts = np.arange(T0, T1 + 0.1, DT); xs = np.linspace(0, 1, NX)
    lab = np.empty((len(Ts), len(xs)), dtype=object)
    for i in range(0, len(Ts), 20):
        a = m.equilibrium(Ts[i:i + 20], xs)
        for ii in range(a.shape[0]):
            for j in range(a.shape[1]):
                c = a[ii, j]
                if not c:
                    c = m.equilibrium_drop_one(Ts[i + ii], xs[j])
                lab[i + ii, j] = "+".join(sorted({pid.bench_class(m.reg_id(n, d)) for n, d in c}))
    np.savez_compressed(cache, T=Ts, x=xs, lab=lab.astype(str))
    return Ts, xs, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grace", default=os.path.join(ROOT, "results/cuzr_grace.json"))
    ap.add_argument("--sgte", default=os.path.join(ROOT, "results/cuzr_sgte_94Zen.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    G, S = json.load(open(a.grace)), json.load(open(a.sgte))
    Ts, xs, lab = phase_map(os.path.join(ROOT, G["tdb"]), G["names_as"], os.path.join(ROOT, "results/map_cuzr_grace.npz"))
    plt.rcParams.update({"font.family": figstyle.FONT,
                         "font.size": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(183 / 25.4, 78 / 25.4))
    ax = fig.add_axes([0.075, 0.13, 0.485, 0.80])
    bx = fig.add_axes([0.70, 0.36, 0.29, 0.57])
    # (a) phase map: fields light, boundaries as contours of each assemblage indicator
    keys = sorted(set(lab.ravel()))
    idx = np.vectorize(keys.index)(lab)
    ax.pcolormesh(xs, Ts, np.zeros_like(idx, float), cmap="Greys", vmin=0, vmax=1, shading="auto")
    one = [k for k in keys if "+" not in k]
    for k in keys:
        ind = (lab == k).astype(float)
        ax.contour(xs, Ts, ind, levels=[0.5], colors=[MUTED], linewidths=0.45)
    for k in keys:   # light tint on one-phase fields (liquid, solid solutions)
        if "+" not in k:
            ax.contourf(xs, Ts, (lab == k).astype(float), levels=[0.5, 1.5], colors=["#e8eef7"])
    for k in keys:   # field labels at the cell of maximum clearance
        ind = lab == k
        if ind.sum() < 60:
            continue
        from scipy.ndimage import distance_transform_edt
        dmap = distance_transform_edt(np.pad(ind, 1))[1:-1, 1:-1]
        i, j = np.unravel_index(np.argmax(dmap), dmap.shape)
        if dmap[i, j] < 4:
            continue
        txt = "+".join(SHORT.get(p, p) for p in k.split("+"))
        ax.text(xs[j], Ts[i], txt, ha="center", va="center", fontsize=4.6, color=MUTED)
    pts = G["points"]
    for kind, mk, ms in (("boundary", "D", 9), ("field", "o", 22)):
        for cat, col, face in (("ok", C_OK, True), ("partial", C_PART, True), ("miss", C_MISS, False)):
            sel = [p for p in pts if p["kind"] == kind and
                   ({"ok": p["recall_point"] == 1, "partial": 0 < p["recall_point"] < 1, "miss": p["recall_point"] == 0}[cat])]
            if not sel:
                continue
            ax.scatter([p["x_Zr"] for p in sel], [p["T_K"] for p in sel], s=ms, marker=mk if face else ("X" if kind == "field" else "x"),
                       facecolors=col, edgecolors="white" if face else col, linewidths=0.4 if face else 0.7, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(T0, T1)
    ax.set_xlabel("x$_{Zr}$ (mole fraction)"); ax.set_ylabel("T (K)")
    gf, gb = G["metrics"]["physical"]["field"]["all"], G["metrics"]["physical"]["boundary"]["all"]
    ax.set_title(f"a  GRACE-2L-OMAT, MLIP-only PhaseForge TDB (no fitting)", loc="left", fontsize=7, fontweight="bold")
    leg = [Line2D([], [], marker="o", ls="", mfc=C_OK, mec="white", ms=4.5, label="field point, phases recovered"),
           Line2D([], [], marker="o", ls="", mfc=C_PART, mec="white", ms=4.5, label="field point, partly"),
           Line2D([], [], marker="X", ls="", mfc=C_MISS, mec=C_MISS, ms=4.5, label="field point, none"),
           Line2D([], [], marker="D", ls="", mfc=C_OK, mec="white", ms=3, label="boundary point (±10 K, ±0.01), recovered"),
           Line2D([], [], marker="D", ls="", mfc=C_PART, mec="white", ms=3, label="boundary point, partly"),
           Line2D([], [], marker="x", ls="", mec=C_MISS, ms=3.5, label="boundary point, none")]
    fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.575, 0.0), fontsize=4.8, frameon=False, ncol=2,
               title="evidence points (colour: reference phases the model recovers)", title_fontsize=4.8, columnspacing=1.0)
    ax.text(0.99, 0.015, f"prevalence-weighted recall: field {gf['weighted']['recall']:.2f} "
            f"[{gf['ci95']['recall'][0]:.2f}, {gf['ci95']['recall'][1]:.2f}], n = {gf['n_points']}; "
            f"boundary {gb['weighted']['recall']:.2f} [{gb['ci95']['recall'][0]:.2f}, {gb['ci95']['recall'][1]:.2f}], n = {gb['n_points']}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8, color=INK,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    # (b) per-phase recall, field + boundary pooled
    def pooled(R):
        out = {}
        for kind in ("field", "boundary"):
            for ph, v in R["metrics"]["physical"][kind]["all"]["per_phase"].items():
                o = out.setdefault(ph, [0, 0]); o[0] += v["tp"]; o[1] += v["support"]
        return {k: (v[0] / v[1] if v[1] else np.nan, v[1]) for k, v in out.items() if v[1]}
    pg, ps = pooled(G), pooled(S)
    phases = sorted(pg, key=lambda k: -pg[k][1])
    y = np.arange(len(phases))
    in_model = set(G["model_classes"]["physical"])
    bx.barh(y - 0.2, [pg[p][0] for p in phases], height=0.38, color=C_GRACE, label="GRACE MLIP-only")
    bx.barh(y + 0.2, [ps.get(p, (0, 0))[0] for p in phases], height=0.38, color=C_SGTE, label="SGTE (Zeng 1994)")
    bx.set_yticks(y)
    bx.set_yticklabels([f"{SHORT.get(p, p)}{'' if p in in_model else ' *'}  ({pg[p][1]})" for p in phases], fontsize=5.2)
    bx.invert_yaxis(); bx.set_xlim(0, 1.0); bx.set_xlabel("recall (field + boundary points)")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.xaxis.grid(True, lw=0.3, color="#d9d9d4"); bx.set_axisbelow(True)
    bx.set_title("b  per-phase recall", loc="left", fontsize=7, fontweight="bold")
    bx.legend(loc="center right", bbox_to_anchor=(1.0, 0.22), fontsize=5, frameon=False)
    bx.text(0.0, -0.29, "(n) reference support; * not in either model's phase list", transform=bx.transAxes,
            fontsize=4.6, color=MUTED)
    figstyle.save(fig, os.path.join(a.out, "fig_cuzr_oneshot"))
    print("wrote", os.path.join(a.out, "fig_cuzr_oneshot.{png,pdf}"))


if __name__ == "__main__":
    main()
