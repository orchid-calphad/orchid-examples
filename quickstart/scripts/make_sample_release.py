"""Build sample-release/: the Cu-Zr slice of an ORCHID release.

The slice has the release's own layout, so that the quickstart runs offline (CI, before the data site is public).
Same files and columns as the release, restricted to system CU-ZR; the verbatim source quotes (`quote` columns) are
left out of this slice.

  python scripts/make_sample_release.py RELEASE_DIR    # a local orchid-data checkout at the pinned commit
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parents[1]
system_id = "CU-ZR"


def main(src: Path) -> None:
    """Write the Cu-Zr slice of the release at src into quickstart/sample-release.

    Args:
        src (Path): Root of a local orchid-data checkout at the pinned commit.
    """
    out = here / "sample-release"
    shutil.rmtree(out, ignore_errors=True)
    (out / "parquet").mkdir(parents=True)
    for t in ("literature_values", "dft_parity"):
        df = pd.read_parquet(src / "parquet" / f"{t}.parquet")
        df = df[df["system"] == system_id].drop(columns=[c for c in df.columns if c == "quote"]).reset_index(drop=True)
        df.to_parquet(out / "parquet" / f"{t}.parquet", index=False, compression="zstd")
    with (src / "espei" / "index.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if r["system"] == system_id]
    (out / "espei").mkdir(parents=True, exist_ok=True)
    with (out / "espei" / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    for r in keep:
        dst = out / "espei" / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "espei" / r["path"], dst)
    shutil.copy(src / "LICENSE", out / "LICENSE")
    print(len(keep), "ESPEI datasets;", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
