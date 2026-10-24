"""The paper's Cu-Zr figure: phase diagrams of the three descriptions (SGTE Zeng 1994, PhaseForge MLIP-only, PhaseForge+
tuned with ORCHID data) overlaid, with the ORCHID phase-equilibrium points; plus a three-panel version.

Geometry from pd_map.py (convex-hull equilibrium on a 1 K x 3,500-point grid, the same Gibbs energies as the TDBs;
validated against pycalphad equilibrium in orchid-phaseforge-demo). Fields coloured: each single-phase field its own
muted colour, two-phase fields a light blend of their two phases; boundaries as continuous polylines; invariants as
horizontal lines between the three phase compositions; line compounds as vertical lines. The gate counts printed on
the figure are read from gate.json (scripts/gate.py), not typed in.

Adapted from orchid-phaseforge-demo scripts/fig_pd.py (paper figure fig_cuzr_overlay): same drawing code; paths, the
font (figstyle.py) and deterministic saving changed, the geometry cache is JSON instead of pickle.

  python scripts/fig_overlay.py [OUT_DIR]     # default figures/; reads OUT_DIR/gate.json or cached/gate.json
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import figstyle as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import model as m, pd_map as pm
ROOT = m.ROOT
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures"
WORK = ROOT / "work"
P = {"SGTE": m.Params.from_tdb(ROOT / "tdb/CuZr-94Zen-LB.tdb"), "MLIP-only": m.Params.from_tdb(ROOT / "tdb/CuZr-PhaseForge-MLIP.tdb"),
     "tuned": m.Params.from_tdb(ROOT / "tdb/CuZr-PhaseForge-final.tdb")}
T0, T1 = 800.0, 1800.0
TS = np.arange(T0, T1 + 0.01, 1.0)
PCOL = {"LIQUID": "#CFE3F1", "FCC_A1": "#F6D2C4", "BCC_A2": "#D3E8C8", "HCP_A3": "#E3D5EC",
        "CU5ZR_C15B": "#E8C27E", "CU51ZR14": "#C9A0C8", "CU8ZR3": "#A9C98F", "CU10ZR7": "#8FB8D6", "CUZR_B2": "#E3A98F", "CUZR2": "#B7A9D6"}
NAME = {"LIQUID": "L", "FCC_A1": "(Cu)", "BCC_A2": "(βZr)", "HCP_A3": "(αZr)", "CU5ZR_C15B": "Cu$_5$Zr", "CU51ZR14": "Cu$_{51}$Zr$_{14}$",
        "CU8ZR3": "Cu$_8$Zr$_3$", "CU10ZR7": "Cu$_{10}$Zr$_7$", "CUZR_B2": "CuZr", "CUZR2": "CuZr$_2$"}
_g = OUT / "gate.json" if (OUT / "gate.json").exists() else ROOT / "cached/gate.json"
GATE = json.load(open(_g))["gate_counts"]


def blend(a, b, w=0.28):
    ca, cb = np.array(to_rgb(PCOL[a])), np.array(to_rgb(PCOL[b]))
    c = 0.5 * (ca + cb)
    return tuple(1 - w * (1 - c))   # lighter blend


def geometry(p, key):
    """Field tracks and compound ranges; cached as JSON in work/ (keyed by the parameter values)."""
    h = hashlib.sha1(repr((sorted(p.L.items()), sorted(p.G.items()), T0, T1)).encode()).hexdigest()[:12]
    c = WORK / f"pdgeo_{key}_{h}.json"
    if c.exists():
        d = json.load(open(c))
        tracks = [{"kind": t["kind"], "phases": tuple(t["phases"]), "T": np.array(t["T"]), "xl": np.array(t["xl"]), "xr": np.array(t["xr"])} for t in d["tracks"]]
        return tracks, {k: tuple(v) for k, v in d["comps"].items()}
    tracks, comps = pm.track(p, TS), pm.compound_ranges(p, TS)
    WORK.mkdir(exist_ok=True)
    json.dump({"tracks": [{"kind": t["kind"], "phases": list(t["phases"]), "T": t["T"].tolist(), "xl": t["xl"].tolist(), "xr": t["xr"].tolist()} for t in tracks],
               "comps": {k: [float(a) for a in v] for k, v in comps.items()}}, open(c, "w"))
    return tracks, comps


def smooth(y, k=5):
    if len(y) < 2 * k + 1: return y
    yp = np.pad(y, k, mode="edge"); w = np.ones(2 * k + 1) / (2 * k + 1)
    ys = np.convolve(yp, w, mode="valid"); ys[:k] = y[:k]; ys[-k:] = y[-k:]   # keep ends exact (invariants)
    return ys


def segments(tr):
    """split a track where the raw boundaries jump (tracking switched region), then smooth each piece."""
    T, xl, xr = tr["T"], tr["xl"], tr["xr"]
    cut = np.where((np.abs(np.diff(xl)) > 0.02) | (np.abs(np.diff(xr)) > 0.02))[0] + 1
    return [(T[a:b], smooth(xl[a:b]), smooth(xr[a:b])) for a, b in zip(np.r_[0, cut], np.r_[cut, len(T)]) if b - a > 1]


def draw(ax, p, geo, *, fill, lw, color, ls="-", alpha=1.0, z=2):
    tracks, comps = geo
    if fill:
        for tr in tracks:
            c = PCOL[tr["phases"][0]] if tr["kind"] == "1" else blend(*tr["phases"])
            for T, xl, xr in segments(tr):   # pad half a grid step at both ends so adjacent fields meet without seams
                TT = np.r_[T[0] - 0.6, T, T[-1] + 0.6]; L_ = np.r_[xl[0], xl, xl[-1]]; R_ = np.r_[xr[0], xr, xr[-1]]
                ax.fill_betweenx(TT, L_, R_, color=c, lw=0, zorder=0.5)
    for tr in tracks:
        segs = segments(tr)
        for si, (T, xl, xr) in enumerate(segs):
            for xe, side in ((xl, 0), (xr, 1)):
                ph = tr["phases"][side if tr["kind"] == "2" else 0]
                if ph in m.COMP: continue
                xe = np.array(xe, float); xe[(xe < 2e-3) | (xe > 1 - 2e-3)] = np.nan
                ax.plot(xe, T, color=color, lw=lw, ls=ls, alpha=alpha, zorder=z, solid_capstyle="round")
        if tr["kind"] == "2" and segs:
            for (T, xl, xr), k in ((segs[0], 0), (segs[-1], -1)):   # invariants at field ends inside the plot
                if T0 + 1 < T[k] < T1 - 1 and xr[k] - xl[k] > 4e-3:
                    ax.plot([xl[k], xr[k]], [T[k], T[k]], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)
    for ph, (x, lo, hi) in comps.items():           # stoichiometric compounds: bold vertical line over their stability range
        ax.plot([x, x], [lo, hi], color=color, lw=lw * 1.5, ls=ls, alpha=alpha, zorder=z + 0.5, solid_capstyle="butt")


def labels(ax, p, geo, fs=6.0, two_phase=True):
    tracks, comps = geo
    halo = [pe.withStroke(linewidth=1.6, foreground="white", alpha=0.8)]
    kw = dict(ha="center", va="center", fontsize=fs, color=st.INK, zorder=8, path_effects=halo)
    placed = []
    def ok(x, T, w=0.06, h=35):
        return all(abs(x - a) > w or abs(T - b) > h for a, b in placed)
    # liquid at top centre
    for tr in tracks:
        if tr["phases"] == ("LIQUID",) and tr["T"][-1] >= T1 - 1:
            ax.text(0.45, T1 - 70, "L", **dict(kw, fontsize=fs + 1)); placed.append((0.45, T1 - 70))
    big = {}
    for tr in tracks:
        area = np.trapezoid(tr["xr"] - tr["xl"], tr["T"]) if len(tr["T"]) > 1 else 0
        key = tr["phases"]
        if key not in big or area > big[key][0]: big[key] = (area, tr)
    for key, (area, tr) in big.items():
        if key == ("LIQUID",): continue
        w = tr["xr"] - tr["xl"]
        if tr["kind"] == "1":
            ph = key[0]
            if ph == "FCC_A1":
                ax.annotate(NAME[ph], xy=(0.003, min(1200, tr["T"].mean())), xytext=(0.06, 900), arrowprops=dict(arrowstyle="-", lw=0.4, color=st.INK2), **kw); continue
            if ph == "HCP_A3":
                ax.annotate(NAME[ph], xy=(0.997, tr["T"][0] + 20), xytext=(0.93, 850), arrowprops=dict(arrowstyle="-", lw=0.4, color=st.INK2), **kw); continue
            if ph == "BCC_A2":
                j = int(np.argmax(w)); x = 0.5 * (tr["xl"][j] + tr["xr"][j])
                if w[j] > 0.04: ax.text(min(x, 0.975), tr["T"][j] - 60 if tr["T"][j] > T1 - 100 else tr["T"][j], NAME[ph], rotation=90, **kw)
                else: ax.annotate(NAME[ph], xy=(0.995, tr["T"].mean()), xytext=(0.95, tr["T"].mean() - 150), arrowprops=dict(arrowstyle="-", lw=0.4, color=st.INK2), **kw)
            continue
        if not two_phase: continue
        hgt = tr["T"][-1] - tr["T"][0]
        Tl = 0.5 * (tr["T"][0] + tr["T"][-1]) if tr["T"][-1] < T1 - 1 else 0.5 * (tr["T"][0] + T1)
        jj = int(np.argmin(abs(tr["T"] - Tl))); xw = tr["xr"][jj] - tr["xl"][jj]; x = 0.5 * (tr["xl"][jj] + tr["xr"][jj])
        a, b = NAME[key[0]], NAME[key[1]]
        if a == "L": a, b = b, a
        if b == "L": a, b = "L", a
        if xw > 0.17 and hgt > 40 and ok(x, Tl):
            ax.text(x, Tl, a + " + " + b, **dict(kw, fontsize=fs - 0.8)); placed.append((x, Tl))
        elif xw > 0.07 and hgt > 110 and ok(x, Tl):
            ax.text(x, Tl, a + "\n+ " + b, **dict(kw, fontsize=fs - 0.8, linespacing=1.0)); placed.append((x, Tl))
    # compound names: horizontal, in a strip above the axes, staggered, with leader lines
    order = sorted(comps.items(), key=lambda kv: kv[1][0])
    for i, (ph, (x, lo, hi)) in enumerate(order):
        yt = 1.035 if i % 2 == 0 else 1.10
        ax.annotate(NAME[ph], xy=(x, T1), xycoords="data", xytext=(x, yt), textcoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=fs - 0.5, color=st.INK, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=0.4, color=st.INK2, shrinkA=0, shrinkB=0))


def zpf_points():
    d = json.load(open(ROOT / "inputs/zpf_points.json"))
    return [(p["x_Zr"], p["T_K"], p["split"], p["kind"]) for p in d["points"]]


def evidence(ax, pts, ms=2.3):
    for x, T, split, kind in pts:
        if kind == "bracket": continue
        if kind == "assessed":
            ax.plot(x, T, "D", ms=1.6, color=st.C_ASSESSED, mew=0, alpha=0.9, zorder=9)
        else:
            ax.plot(x, T, "o", ms=ms, mfc=(st.C_EXP if split == "fit" else "white"), mec=st.C_EXP, mew=0.5, alpha=0.75, zorder=9)


def axes_style(ax, left=True, right=True):
    ax.set_xlim(0, 1); ax.set_ylim(T0, T1); ax.set_xlabel("x(Zr)")
    ax.tick_params(direction="out")
    if left: ax.set_ylabel("Temperature (K)")
    if right:
        sec = ax.secondary_yaxis("right", functions=(lambda T: T - 273.15, lambda c: c + 273.15))
        sec.set_ylabel("Temperature (°C)"); sec.tick_params(direction="out", labelsize=6.5)
        sec.spines["right"].set_linewidth(0.5)


def legend_items():
    return [Line2D([], [], color=st.ACCENT, lw=1.2, label="PhaseForge+ tuned"), Line2D([], [], color=st.INK2, lw=0.7, label="SGTE (Zeng 1994)"),
            Line2D([], [], color="#8A7FA8", lw=0.6, ls=(0, (3, 1.5)), label="PhaseForge MLIP-only"),
            Line2D([], [], ls="", marker="o", ms=2.5, mfc=st.C_EXP, mec=st.C_EXP, label="ORCHID ZPF (fitted)"),
            Line2D([], [], ls="", marker="o", ms=2.5, mfc="white", mec=st.C_EXP, label="ORCHID ZPF (held out)"),
            Line2D([], [], ls="", marker="D", ms=2.0, color=st.C_ASSESSED, label="Assessed-diagram field points")]


def main():
    geo = {k: geometry(p, k) for k, p in P.items()}
    pts = zpf_points()
    # (1) overlay (the paper figure)
    fig, ax = plt.subplots(figsize=(120 * st.MM, 108 * st.MM), constrained_layout=True)
    draw(ax, P["tuned"], geo["tuned"], fill=True, lw=1.2, color=st.ACCENT, z=4)
    draw(ax, P["SGTE"], geo["SGTE"], fill=False, lw=0.7, color=st.INK2, z=3)
    draw(ax, P["MLIP-only"], geo["MLIP-only"], fill=False, lw=0.6, color="#8A7FA8", ls=(0, (3, 1.5)), alpha=0.9, z=2)
    labels(ax, P["tuned"], geo["tuned"])
    evidence(ax, pts)
    axes_style(ax)
    fig.legend(handles=legend_items(), loc="outside lower center", ncol=3, fontsize=5.5, frameon=False, columnspacing=1.2, handlelength=2.2)
    ax.text(0.015, 0.985, "Invariants within the gate: MLIP-only %d/%d, tuned %d/%d (in-sample), SGTE %d/%d" % (*GATE["MLIP-only"], *GATE["tuned"], *GATE["SGTE"]),
            transform=ax.transAxes, ha="left", va="top", fontsize=5, color=st.INK2, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
    st.save(fig, OUT / "fig_cuzr_overlay")
    # (2) three panels
    fig, axs = plt.subplots(1, 3, figsize=(st.W2, 95 * st.MM), sharey=True, constrained_layout=True)
    for i, (k, title) in enumerate((("SGTE", "SGTE (Zeng 1994)"), ("MLIP-only", "PhaseForge MLIP-only"), ("tuned", "PhaseForge+ tuned with ORCHID data"))):
        ax = axs[i]
        draw(ax, P[k], geo[k], fill=True, lw=0.9 if k != "tuned" else 1.1, color=st.ACCENT if k == "tuned" else st.INK, z=4)
        labels(ax, P[k], geo[k], fs=5.5, two_phase=False)
        evidence(ax, pts, ms=2.0)
        axes_style(ax, left=(i == 0), right=(i == 2))
        ax.set_title(title + ("\ninvariants within the gate: %d/%d" % tuple(GATE[k])) + (" (in-sample)" if k == "tuned" else ""), fontsize=6.5, loc="center", pad=28, linespacing=1.3)
    fig.legend(handles=legend_items()[3:], loc="outside lower center", ncol=3, fontsize=5.5, frameon=False)
    st.save(fig, OUT / "fig_cuzr_panels")
    print("wrote", OUT / "fig_cuzr_overlay.{png,pdf}", OUT / "fig_cuzr_panels.{png,pdf}")


if __name__ == "__main__":
    main()
