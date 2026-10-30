"""The paper's Cu-Zr figure: the three phase diagrams overlaid, and a three-panel version.

The descriptions are SGTE Zeng 1994, PhaseForge MLIP-only and PhaseForge+ tuned with ORCHID data. The ORCHID
phase-equilibrium points are drawn on both.

Geometry from pd_map.py (convex-hull equilibrium on a 1 K x 3,500-point grid, the same Gibbs energies as the TDBs;
validated against pycalphad equilibrium in orchid-phaseforge-demo). Fields coloured: each single-phase field its own
muted colour, two-phase fields a light blend of their two phases; boundaries as continuous polylines; invariants as
horizontal lines between the three phase compositions; line compounds as vertical lines. The gate counts printed on
the figure are read from gate.json (scripts/gate.py), not typed in.

Adapted from orchid-phaseforge-demo scripts/fig_pd.py (paper figure fig_cuzr_overlay): same drawing code; paths, the
font (figstyle.py) and deterministic saving changed, the geometry cache is JSON instead of pickle.

  python scripts/fig_overlay.py [OUT_DIR]     # default figures/; reads OUT_DIR/gate.json or cached/gate.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # noqa: E402 - pd_map.py and model.py sit beside this script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # noqa: E402 - figstyle.py is shared
import figstyle as st  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import model as m  # noqa: E402
import pd_map as pm  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

root = m.root
work_dir = root / "work"
t_lo = 800.0  # K, lower end of the plotted temperature range
t_hi = 1800.0  # K, upper end of the plotted temperature range
t_grid = np.arange(t_lo, t_hi + 0.01, 1.0)  # 1 K grid
jump_tol = 0.02  # mole fraction; a jump of a boundary larger than this starts a new track segment
edge_eps = 2e-3  # mole fraction; boundaries this close to 0 or 1 are not drawn
min_invariant_width = 4e-3  # mole fraction; field ends narrower than this get no invariant line
phase_colors = {
    "LIQUID": "#CFE3F1",
    "FCC_A1": "#F6D2C4",
    "BCC_A2": "#D3E8C8",
    "HCP_A3": "#E3D5EC",
    "CU5ZR_C15B": "#E8C27E",
    "CU51ZR14": "#C9A0C8",
    "CU8ZR3": "#A9C98F",
    "CU10ZR7": "#8FB8D6",
    "CUZR_B2": "#E3A98F",
    "CUZR2": "#B7A9D6",
}
phase_names = {
    "LIQUID": "L",
    "FCC_A1": "(Cu)",
    "BCC_A2": "(βZr)",
    "HCP_A3": "(αZr)",
    "CU5ZR_C15B": "Cu$_5$Zr",
    "CU51ZR14": "Cu$_{51}$Zr$_{14}$",
    "CU8ZR3": "Cu$_8$Zr$_3$",
    "CU10ZR7": "Cu$_{10}$Zr$_7$",
    "CUZR_B2": "CuZr",
    "CUZR2": "CuZr$_2$",
}


def blend(a, b, w=0.28):
    """Light blend of two phase colours, used for two-phase fields.

    Args:
        a (str): First phase name, a key of phase_colors.
        b (str): Second phase name, a key of phase_colors.
        w (float): Weight of the white mixed in (0 keeps the average colour).

    Returns:
        tuple[float, float, float]: RGB colour with components in [0, 1].
    """
    ca, cb = np.array(to_rgb(phase_colors[a])), np.array(to_rgb(phase_colors[b]))
    c = 0.5 * (ca + cb)
    return tuple(1 - w * (1 - c))  # lighter blend


def geometry(p, key):
    """Field tracks and compound ranges of one parameter set, cached as JSON in work/ (keyed by parameter values).

    Args:
        p (m.Params): Parameter set of the TDB.
        key (str): Description name, used in the cache file name.

    Returns:
        tuple[list[dict], dict[str, tuple[float, float, float]]]: Tracks (keys "kind", "phases", arrays "T", "xl",
        "xr") and compound ranges (x_Zr, T_min, T_max in K).
    """
    h = hashlib.sha1(repr((sorted(p.L.items()), sorted(p.G.items()), t_lo, t_hi)).encode()).hexdigest()[:12]
    c = work_dir / f"pdgeo_{key}_{h}.json"
    if c.exists():
        d = json.loads(c.read_text(encoding="utf-8"))
        tracks = [
            {
                "kind": t["kind"],
                "phases": tuple(t["phases"]),
                "T": np.array(t["T"]),
                "xl": np.array(t["xl"]),
                "xr": np.array(t["xr"]),
            }
            for t in d["tracks"]
        ]
        return tracks, {k: tuple(v) for k, v in d["comps"].items()}
    tracks, comps = pm.track(p, t_grid), pm.compound_ranges(p, t_grid)
    work_dir.mkdir(exist_ok=True)
    c.write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "kind": t["kind"],
                        "phases": list(t["phases"]),
                        "T": t["T"].tolist(),
                        "xl": t["xl"].tolist(),
                        "xr": t["xr"].tolist(),
                    }
                    for t in tracks
                ],
                "comps": {k: [float(a) for a in v] for k, v in comps.items()},
            }
        ),
        encoding="utf-8",
    )
    return tracks, comps


def smooth(y, k=5):
    """Centred moving average of a boundary curve, with its end points kept exact.

    Args:
        y (np.ndarray): Boundary values along a track.
        k (int): Half-width of the window in samples.

    Returns:
        np.ndarray: Smoothed values; the first and last k entries equal the input (invariant end points).
    """
    if len(y) < 2 * k + 1:
        return y
    yp = np.pad(y, k, mode="edge")
    w = np.ones(2 * k + 1) / (2 * k + 1)
    ys = np.convolve(yp, w, mode="valid")
    ys[:k] = y[:k]  # keep ends exact (invariants)
    ys[-k:] = y[-k:]
    return ys


def segments(tr):
    """Split a track where the raw boundaries jump (the tracking switched region), then smooth each piece.

    Args:
        tr (dict): Track with arrays "T", "xl" and "xr".

    Returns:
        list[tuple[np.ndarray, np.ndarray, np.ndarray]]: (T, smoothed x_left, smoothed x_right) per piece with more
        than one point.
    """
    t_axis, xl, xr = tr["T"], tr["xl"], tr["xr"]
    cut = np.where((np.abs(np.diff(xl)) > jump_tol) | (np.abs(np.diff(xr)) > jump_tol))[0] + 1
    bounds = zip(np.r_[0, cut], np.r_[cut, len(t_axis)], strict=True)
    return [(t_axis[a:b], smooth(xl[a:b]), smooth(xr[a:b])) for a, b in bounds if b - a > 1]


def draw(ax, p, geo, *, fill, lw, color, ls="-", alpha=1.0, z=2):
    """Draw one description's phase diagram: optional field fills, boundaries, invariants and compound lines.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        p (m.Params): Parameter set of the TDB (not used for drawing; kept for the call signature).
        geo (tuple): Output of geometry() for p.
        fill (bool): Whether to fill the fields with their colours.
        lw (float): Line width in pt.
        color (str): Line colour.
        ls (str | tuple): Line style.
        alpha (float): Line opacity.
        z (float): Drawing order of the lines.
    """
    tracks, comps = geo
    if fill:
        for tr in tracks:
            c = phase_colors[tr["phases"][0]] if tr["kind"] == "1" else blend(*tr["phases"])
            for t_axis, xl, xr in segments(tr):  # pad half a grid step at both ends so adjacent fields meet without seams
                t_pad = np.r_[t_axis[0] - 0.6, t_axis, t_axis[-1] + 0.6]
                x_left = np.r_[xl[0], xl, xl[-1]]
                x_right = np.r_[xr[0], xr, xr[-1]]
                ax.fill_betweenx(t_pad, x_left, x_right, color=c, lw=0, zorder=0.5)
    for tr in tracks:
        segs = segments(tr)
        for t_axis, xl, xr in segs:
            for x_edge, side in ((xl, 0), (xr, 1)):
                ph = tr["phases"][side if tr["kind"] == "2" else 0]
                if ph in m.line_compounds:
                    continue
                edge = np.array(x_edge, float)
                edge[(edge < edge_eps) | (edge > 1 - edge_eps)] = np.nan
                ax.plot(edge, t_axis, color=color, lw=lw, ls=ls, alpha=alpha, zorder=z, solid_capstyle="round")
        if tr["kind"] == "2" and segs:
            for (t_axis, xl, xr), k in ((segs[0], 0), (segs[-1], -1)):  # invariants at field ends inside the plot
                if t_lo + 1 < t_axis[k] < t_hi - 1 and xr[k] - xl[k] > min_invariant_width:
                    ax.plot([xl[k], xr[k]], [t_axis[k], t_axis[k]], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)
    for x, lo, hi in comps.values():  # stoichiometric compounds: bold vertical line over their stability range
        ax.plot([x, x], [lo, hi], color=color, lw=lw * 1.5, ls=ls, alpha=alpha, zorder=z + 0.5, solid_capstyle="butt")


def _label_single_phase(ax, ph, tr, w, kw) -> None:
    """Label one single-phase solution field (FCC, HCP or BCC); other phases get no label.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        ph (str): Phase name of the field.
        tr (dict): Track of the field.
        w (np.ndarray): Field width in mole fraction along the track.
        kw (dict): Text keyword arguments shared by all labels.
    """
    arrow = {"arrowstyle": "-", "lw": 0.4, "color": st.ink2}
    if ph == "FCC_A1":
        ax.annotate(phase_names[ph], xy=(0.003, min(1200, tr["T"].mean())), xytext=(0.06, 900), arrowprops=arrow, **kw)
    elif ph == "HCP_A3":
        ax.annotate(phase_names[ph], xy=(0.997, tr["T"][0] + 20), xytext=(0.93, 850), arrowprops=arrow, **kw)
    elif ph == "BCC_A2":
        j = int(np.argmax(w))
        x = 0.5 * (tr["xl"][j] + tr["xr"][j])
        if w[j] > 0.04:
            ax.text(
                min(x, 0.975),
                tr["T"][j] - 60 if tr["T"][j] > t_hi - 100 else tr["T"][j],
                phase_names[ph],
                rotation=90,
                **kw,
            )
        else:
            ax.annotate(phase_names[ph], xy=(0.995, tr["T"].mean()), xytext=(0.95, tr["T"].mean() - 150), arrowprops=arrow, **kw)


def _label_text_kwargs(fs) -> dict:
    """Keyword arguments shared by all phase-name labels: centred, dark, with a white halo.

    Args:
        fs (float): Font size in pt.

    Returns:
        dict: Keyword arguments for Axes.text and Axes.annotate.
    """
    halo = [pe.withStroke(linewidth=1.6, foreground="white", alpha=0.8)]
    return {"ha": "center", "va": "center", "fontsize": fs, "color": st.ink, "zorder": 8, "path_effects": halo}


def _is_clear(x, temp, placed, w=0.06, h=35) -> bool:
    """Whether a label centred at (x, temp) avoids every label already placed.

    Args:
        x (float): Zr mole fraction of the label centre.
        temp (float): Temperature of the label centre in K.
        placed (list[tuple[float, float]]): Centres of the labels already placed.
        w (float): Horizontal clearance in mole fraction.
        h (float): Vertical clearance in K.

    Returns:
        bool: True when no placed label is within both clearances.
    """
    return all(abs(x - a) > w or abs(temp - b) > h for a, b in placed)


def _label_two_phase(ax, key, tr, fs, placed) -> None:
    """Label one two-phase field with its phase pair when there is room, and record the label position.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        key (tuple[str, str]): Phase pair of the field.
        tr (dict): Track of the field.
        fs (float): Base font size in pt.
        placed (list[tuple[float, float]]): Label positions already used; extended in place.
    """
    kw = _label_text_kwargs(fs)
    hgt = tr["T"][-1] - tr["T"][0]
    tl = 0.5 * (tr["T"][0] + tr["T"][-1]) if tr["T"][-1] < t_hi - 1 else 0.5 * (tr["T"][0] + t_hi)
    jj = int(np.argmin(abs(tr["T"] - tl)))
    xw = tr["xr"][jj] - tr["xl"][jj]
    x = 0.5 * (tr["xl"][jj] + tr["xr"][jj])
    a, b = phase_names[key[0]], phase_names[key[1]]
    if a == "L":
        a, b = b, a
    if b == "L":
        a, b = "L", a
    if xw > 0.17 and hgt > 40 and _is_clear(x, tl, placed):
        ax.text(x, tl, a + " + " + b, **{**kw, "fontsize": fs - 0.8})
        placed.append((x, tl))
    elif xw > 0.07 and hgt > 110 and _is_clear(x, tl, placed):
        ax.text(x, tl, a + "\n+ " + b, **{**kw, "fontsize": fs - 0.8, "linespacing": 1.0})
        placed.append((x, tl))


def _label_compounds(ax, comps, fs) -> None:
    """Name the line compounds in a staggered strip above the axes, with leader lines.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        comps (dict[str, tuple[float, float, float]]): Compound ranges from geometry().
        fs (float): Base font size in pt.
    """
    order = sorted(comps.items(), key=lambda kv: kv[1][0])
    for i, (ph, (x, _lo, _hi)) in enumerate(order):
        yt = 1.035 if i % 2 == 0 else 1.10
        ax.annotate(
            phase_names[ph],
            xy=(x, t_hi),
            xycoords="data",
            xytext=(x, yt),
            textcoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=fs - 0.5,
            color=st.ink,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": st.ink2, "shrinkA": 0, "shrinkB": 0},
        )


def labels(ax, p, geo, fs=6.0, two_phase=True):
    """Place phase names for single-phase fields, two-phase fields and compounds on the axes.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        p (m.Params): Parameter set of the TDB (not used for placement; kept for the call signature).
        geo (tuple): Output of geometry() for p.
        fs (float): Base font size in pt.
        two_phase (bool): Whether to label the two-phase fields.
    """
    tracks, comps = geo
    kw = _label_text_kwargs(fs)
    placed = []
    # liquid at top centre
    for tr in tracks:
        if tr["phases"] == ("LIQUID",) and tr["T"][-1] >= t_hi - 1:
            ax.text(0.45, t_hi - 70, "L", **{**kw, "fontsize": fs + 1})
            placed.append((0.45, t_hi - 70))
    big = {}  # the widest track of each phase set carries its label
    for tr in tracks:
        area = np.trapezoid(tr["xr"] - tr["xl"], tr["T"]) if len(tr["T"]) > 1 else 0
        key = tr["phases"]
        if key not in big or area > big[key][0]:
            big[key] = (area, tr)
    for key, (_area, tr) in big.items():
        if key == ("LIQUID",):
            continue
        if tr["kind"] == "1":
            _label_single_phase(ax, key[0], tr, tr["xr"] - tr["xl"], kw)
        elif two_phase:
            _label_two_phase(ax, key, tr, fs, placed)
    _label_compounds(ax, comps, fs)


def zpf_points():
    """Phase-equilibrium points drawn as markers, read from inputs/zpf_points.json.

    Returns:
        list[tuple[float, float, str, str]]: (x_Zr, T in K, split, kind) per point.
    """
    d = json.loads((root / "inputs/zpf_points.json").read_text(encoding="utf-8"))
    return [(p["x_Zr"], p["T_K"], p["split"], p["kind"]) for p in d["points"]]


def evidence(ax, pts, ms=2.3):
    """Draw the ORCHID points: filled circles for fitted, open for held-out, diamonds for assessed-diagram points.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        pts (list[tuple[float, float, str, str]]): Points from zpf_points().
        ms (float): Marker size of the circles in pt.
    """
    for x, temp, split, kind in pts:
        if kind == "bracket":
            continue
        if kind == "assessed":
            ax.plot(x, temp, "D", ms=1.6, color=st.c_assessed, mew=0, alpha=0.9, zorder=9)
        else:
            ax.plot(
                x, temp, "o", ms=ms, mfc=(st.c_exp if split == "fit" else "white"), mec=st.c_exp, mew=0.5, alpha=0.75, zorder=9
            )


def axes_style(ax, left=True, right=True):
    """Set the axis limits, labels and the Celsius secondary axis.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        left (bool): Whether to label the left y axis.
        right (bool): Whether to add the Celsius axis on the right.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(t_lo, t_hi)
    ax.set_xlabel("x(Zr)")
    ax.tick_params(direction="out")
    if left:
        ax.set_ylabel("Temperature (K)")
    if right:
        sec = ax.secondary_yaxis("right", functions=(lambda t: t - 273.15, lambda c: c + 273.15))
        sec.set_ylabel("Temperature (°C)")
        sec.tick_params(direction="out", labelsize=6.5)
        sec.spines["right"].set_linewidth(0.5)


def legend_items():
    """Legend handles for the description lines and the ORCHID markers.

    Returns:
        list[Line2D]: Legend entries in display order.
    """
    return [
        Line2D([], [], color=st.accent, lw=1.2, label="PhaseForge+ tuned"),
        Line2D([], [], color=st.ink2, lw=0.7, label="SGTE (Zeng 1994)"),
        Line2D([], [], color="#8A7FA8", lw=0.6, ls=(0, (3, 1.5)), label="PhaseForge MLIP-only"),
        Line2D([], [], ls="", marker="o", ms=2.5, mfc=st.c_exp, mec=st.c_exp, label="ORCHID ZPF (fitted)"),
        Line2D([], [], ls="", marker="o", ms=2.5, mfc="white", mec=st.c_exp, label="ORCHID ZPF (held out)"),
        Line2D([], [], ls="", marker="D", ms=2.0, color=st.c_assessed, label="Assessed-diagram field points"),
    ]


def _overlay_figure(out, params, geo, pts, gate_counts) -> None:
    """Write the paper figure: the three descriptions on one diagram, with the gate counts in the corner.

    Args:
        out (Path): Output directory.
        params (dict[str, m.Params]): Parameter sets by description name.
        geo (dict[str, tuple]): Geometry per description from geometry().
        pts (list[tuple[float, float, str, str]]): ORCHID points from zpf_points().
        gate_counts (dict[str, list[int]]): Gate counts [passed, total] per description.
    """
    fig, ax = plt.subplots(figsize=(120 * st.mm, 108 * st.mm), constrained_layout=True)
    draw(ax, params["tuned"], geo["tuned"], fill=True, lw=1.2, color=st.accent, z=4)
    draw(ax, params["SGTE"], geo["SGTE"], fill=False, lw=0.7, color=st.ink2, z=3)
    draw(ax, params["MLIP-only"], geo["MLIP-only"], fill=False, lw=0.6, color="#8A7FA8", ls=(0, (3, 1.5)), alpha=0.9, z=2)
    labels(ax, params["tuned"], geo["tuned"])
    evidence(ax, pts)
    axes_style(ax)
    fig.legend(
        handles=legend_items(),
        loc="outside lower center",
        ncol=3,
        fontsize=5.5,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
    )
    mlip_a, mlip_b = gate_counts["MLIP-only"]
    tuned_a, tuned_b = gate_counts["tuned"]
    sgte_a, sgte_b = gate_counts["SGTE"]
    ax.text(
        0.015,
        0.985,
        f"Invariants within the gate: MLIP-only {mlip_a}/{mlip_b}, tuned {tuned_a}/{tuned_b} (in-sample), SGTE {sgte_a}/{sgte_b}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5,
        color=st.ink2,
        bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.0},
    )
    st.save(fig, out / "fig_cuzr_overlay")


def _panel_figure(out, params, geo, pts, gate_counts) -> None:
    """Write the three-panel figure: one diagram per description, each with its own gate count.

    Args:
        out (Path): Output directory.
        params (dict[str, m.Params]): Parameter sets by description name.
        geo (dict[str, tuple]): Geometry per description from geometry().
        pts (list[tuple[float, float, str, str]]): ORCHID points from zpf_points().
        gate_counts (dict[str, list[int]]): Gate counts [passed, total] per description.
    """
    fig, axs = plt.subplots(1, 3, figsize=(st.w2, 95 * st.mm), sharey=True, constrained_layout=True)
    panels = (
        ("SGTE", "SGTE (Zeng 1994)"),
        ("MLIP-only", "PhaseForge MLIP-only"),
        ("tuned", "PhaseForge+ tuned with ORCHID data"),
    )
    for i, (k, title) in enumerate(panels):
        ax = axs[i]
        draw(ax, params[k], geo[k], fill=True, lw=0.9 if k != "tuned" else 1.1, color=st.accent if k == "tuned" else st.ink, z=4)
        labels(ax, params[k], geo[k], fs=5.5, two_phase=False)
        evidence(ax, pts, ms=2.0)
        axes_style(ax, left=(i == 0), right=(i == 2))
        count_a, count_b = gate_counts[k]
        subtitle = f"\ninvariants within the gate: {count_a}/{count_b}" + (" (in-sample)" if k == "tuned" else "")
        ax.set_title(title + subtitle, fontsize=6.5, loc="center", pad=28, linespacing=1.3)
    fig.legend(handles=legend_items()[3:], loc="outside lower center", ncol=3, fontsize=5.5, frameon=False)
    st.save(fig, out / "fig_cuzr_panels")


def main() -> None:
    """Write the overlay figure and the three-panel figure as PNG and PDF into the output directory."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "figures"
    gate_path = out / "gate.json" if (out / "gate.json").exists() else root / "cached/gate.json"
    gate_counts = json.loads(gate_path.read_text(encoding="utf-8"))["gate_counts"]
    params = {
        "SGTE": m.Params.from_tdb(root / "tdb/CuZr-94Zen-LB.tdb"),
        "MLIP-only": m.Params.from_tdb(root / "tdb/CuZr-PhaseForge-MLIP.tdb"),
        "tuned": m.Params.from_tdb(root / "tdb/CuZr-PhaseForge-final.tdb"),
    }
    geo = {k: geometry(p, k) for k, p in params.items()}
    pts = zpf_points()
    _overlay_figure(out, params, geo, pts, gate_counts)
    _panel_figure(out, params, geo, pts, gate_counts)
    print("wrote", out / "fig_cuzr_overlay.{png,pdf}", out / "fig_cuzr_panels.{png,pdf}")


if __name__ == "__main__":
    main()
