"""Write a phase-map cache as a GRID.csv for score_grid.py.

The cache is results/map_cuzr_grace.npz (T, x, labels joined by '+'). Used by the tests to check the grid path against the TDB path; also a template for the CSV format.
  python npz_to_grid.py results/map_cuzr_grace.npz grid.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    """Convert the phase-map cache to the grid CSV read by score_grid.py."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("csv")
    a = ap.parse_args()
    with np.load(a.npz, allow_pickle=False) as z:
        temps, xs, lab = z["T"], z["x"], z["lab"]
        with Path(a.csv).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["T_K", "x_Zr", "phases"])
            for i, temp in enumerate(temps):
                for j, x in enumerate(xs):
                    w.writerow([f"{temp:.1f}", f"{x:.4f}", str(lab[i, j]).replace("+", ";")])


if __name__ == "__main__":
    main()
