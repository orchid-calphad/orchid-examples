"""Figure style shared by the examples: the ORCHID paper palette (Okabe-Ito, fixed meanings) with a font that ships
with matplotlib (DejaVu Sans), so that a figure regenerates byte-identically on any machine with the pinned
matplotlib. The paper's own figures use Arial; set ORCHID_FIG_FONT=Arial to match them (hashes then differ).

``save(fig, stem)`` writes <stem>.png (300 dpi) and <stem>.pdf with fixed metadata (no creation date), which makes
both outputs deterministic."""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
W1, W2, HMAX = 89 * MM, 183 * MM, 170 * MM
C_EXP = "#0072B2"        # experimental
C_MODEL = "#D55E00"      # model
C_MP = "#009E73"         # DFT, Materials Project
C_OQMD = "#56B4E9"       # DFT, OQMD
C_LITDFT = "#CC79A7"     # DFT printed in the literature
C_ASSESSED = "#E69F00"   # assessed phase diagram (handbook)
C_UNSPEC = "#999999"
ACCENT = "#7B3FA0"       # ORCHID accent: tuned / refined model
INK, INK2 = "#000000", "#4D4D4D"
GREY_L, GREY_M = "#E8E8E8", "#BDBDBD"
SRC = {"mp": C_MP, "oqmd": C_OQMD, "literature": C_LITDFT}
SRCL = {"mp": "Materials Project", "oqmd": "OQMD", "literature": "Literature DFT"}
UNIT_KJA = r"kJ (mol of atoms)$^{-1}$"
FONT = os.environ.get("ORCHID_FIG_FONT", "DejaVu Sans")


def apply():
    matplotlib.rcParams.update({
        "font.family": FONT, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.8, "patch.linewidth": 0.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular", "svg.hashsalt": "orchid",
        "path.simplify": True})


def panel_label(ax, s, x=-0.16, y=1.0):
    """Bold panel label at the top-left, outside the axes (axes-fraction coordinates)."""
    ax.text(x, y, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def save(fig, stem, dpi=300):
    """Write <stem>.png and <stem>.pdf deterministically; returns the two paths."""
    stem = str(stem)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(stem + ".png", dpi=dpi, metadata={"Software": None})
    fig.savefig(stem + ".pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return stem + ".png", stem + ".pdf"


apply()
