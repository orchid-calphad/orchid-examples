"""Build sample-release/: the Cu-Zr slice of an ORCHID release, in the release's own layout, so that the quickstart runs
offline (CI, before the data site is public). Same files and columns as the release, restricted to system CU-ZR;
the verbatim source quotes (`quote` columns) are left out of this slice.

  python scripts/make_sample_release.py RELEASE_DIR    # a local orchid-data checkout at the pinned commit
"""
import csv, io, json, os, shutil, sys
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, OUT, SYS = sys.argv[1], os.path.join(HERE, "sample-release"), "CU-ZR"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "parquet"))
for t in ("literature_values", "dft_parity"):
    df = pd.read_parquet(os.path.join(SRC, "parquet", f"{t}.parquet"))
    df = df[df["system"] == SYS].drop(columns=[c for c in df.columns if c == "quote"]).reset_index(drop=True)
    df.to_parquet(os.path.join(OUT, "parquet", f"{t}.parquet"), index=False, compression="zstd")
rows = list(csv.DictReader(open(os.path.join(SRC, "espei", "index.csv"))))
keep = [r for r in rows if r["system"] == SYS]
with open(os.path.join(OUT, "espei", "index.csv") if os.makedirs(os.path.join(OUT, "espei"), exist_ok=True) is None else "", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(keep)
for r in keep:
    dst = os.path.join(OUT, "espei", r["path"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(SRC, "espei", r["path"]), dst)
shutil.copy(os.path.join(SRC, "LICENSE"), os.path.join(OUT, "LICENSE"))
print(len(keep), "ESPEI datasets;", OUT)
