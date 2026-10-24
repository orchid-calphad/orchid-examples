"""Write a phase-map cache (results/map_cuzr_grace.npz: T, x, labels joined by '+') as a GRID.csv for score_grid.py.
Used by the tests to check the grid path against the TDB path; also a template for the CSV format.
  python npz_to_grid.py results/map_cuzr_grace.npz grid.csv"""
import csv, sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=False)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["T_K", "x_Zr", "phases"])
    for i, T in enumerate(z["T"]):
        for j, x in enumerate(z["x"]):
            w.writerow([f"{T:.1f}", f"{x:.4f}", str(z["lab"][i, j]).replace("+", ";")])
