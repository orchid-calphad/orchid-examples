"""Figure style shared by the examples.

The ORCHID paper palette (Okabe-Ito, fixed meanings) with a font that ships with matplotlib (DejaVu Sans), so that a
figure regenerates byte-identically on any machine with the pinned matplotlib. The paper's own figures use Arial; set
``ORCHID_FIG_FONT=Arial`` to match them (hashes then differ).

``save(fig, stem)`` writes ``<stem>.png`` (300 dpi) and ``<stem>.pdf`` with fixed metadata (no creation date), which
makes both outputs deterministic.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

mm = 1 / 25.4  # inches per millimetre
w1 = 89 * mm  # single-column figure width, inches
w2 = 183 * mm  # double-column figure width, inches
hmax = 170 * mm  # maximum figure height, inches

c_exp = "#0072B2"  # experimental
c_model = "#D55E00"  # model
c_mp = "#009E73"  # DFT, Materials Project
c_oqmd = "#56B4E9"  # DFT, OQMD
c_litdft = "#CC79A7"  # DFT printed in the literature
c_assessed = "#E69F00"  # assessed phase diagram (handbook)
c_unspec = "#999999"  # unspecified source
accent = "#7B3FA0"  # ORCHID accent: tuned / refined model
ink = "#000000"
ink2 = "#4D4D4D"
grey_l = "#E8E8E8"
grey_m = "#BDBDBD"
src = {"mp": c_mp, "oqmd": c_oqmd, "literature": c_litdft}
src_label = {"mp": "Materials Project", "oqmd": "OQMD", "literature": "Literature DFT"}
unit_kja = r"kJ (mol of atoms)$^{-1}$"
font = os.environ.get("ORCHID_FIG_FONT", "DejaVu Sans")

png_dpi = 300  # resolution of the raster output of save()
rc_font_size = 7  # pt, text of titles and axis labels
rc_tick_size = 6.5  # pt, tick labels and legend
rc_axis_width = 0.5  # pt, axes and tick line widths
rc_line_width = 0.8  # pt, plotted lines
rc_tick_length = 2  # pt, major tick length


def apply() -> None:
    """Set the matplotlib rcParams of the ORCHID figure style (process-wide, applied on import)."""
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": rc_font_size,
            "axes.titlesize": rc_font_size,
            "axes.labelsize": rc_font_size,
            "xtick.labelsize": rc_tick_size,
            "ytick.labelsize": rc_tick_size,
            "legend.fontsize": rc_tick_size,
            "legend.frameon": False,
            "axes.linewidth": rc_axis_width,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": rc_tick_length,
            "ytick.major.size": rc_tick_length,
            "xtick.major.width": rc_axis_width,
            "ytick.major.width": rc_axis_width,
            "lines.linewidth": rc_line_width,
            "patch.linewidth": rc_axis_width,
            "axes.edgecolor": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "text.color": ink,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
            "svg.hashsalt": "orchid",
            "path.simplify": True,
        }
    )


def panel_label(ax: plt.Axes, s: str, x: float = -0.16, y: float = 1.0) -> None:
    """Write a bold panel label at the top-left corner of an axes, outside the plot area.

    Args:
        ax (plt.Axes): Axes to label.
        s (str): Label text, for example ``"(a)"``.
        x (float): Horizontal position in axes-fraction coordinates (0 is the left edge).
        y (float): Vertical position in axes-fraction coordinates (1 is the top edge).
    """
    ax.text(x, y, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def save(fig: plt.Figure, stem: str | Path, dpi: int = png_dpi) -> tuple[str, str]:
    """Write ``<stem>.png`` and ``<stem>.pdf`` deterministically and close the figure.

    Args:
        fig (plt.Figure): Figure to write.
        stem (str | Path): Output path without extension; parent directories are created.
        dpi (int): Raster resolution of the PNG, in dots per inch.

    Returns:
        tuple[str, str]: Paths of the PNG and the PDF.
    """
    stem = str(stem)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem + ".png", dpi=dpi, metadata={"Software": None})
    fig.savefig(stem + ".pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return stem + ".png", stem + ".pdf"


apply()
