"""Run the quick path of each example and check that its outputs regenerate identically.

Each example's ``run_quick.sh OUT`` writes its outputs into a temporary directory. Every output listed in
``tests/examples.json`` must equal the committed file (sha256). With ``--pixel-tol T``, a PNG whose bytes differ passes
when the RMS difference of its pixels (0-255 scale) is at most T (fonts and rasterisation differ across platforms);
the PDF and JSON outputs of such an example are then reported but not failed unless ``--strict``.

Usage::

    python tests/check_quick_paths.py [EXAMPLE ...] [--pixel-tol 0] [--strict] [--keep]

Uses the current interpreter (``PYTHON``) for the examples' scripts. Exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from matplotlib import image

ROOT = Path(__file__).resolve().parent.parent
rgb_scale = 255.0  # pixel values are compared on the 0-255 scale


def sha(p: Path) -> str:
    """Return the sha256 hex digest of a file.

    Args:
        p (Path): File to hash.

    Returns:
        str: Lower-case hexadecimal digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rms(a: Path, b: Path) -> float:
    """Return the RMS pixel difference of two PNG images on the 0-255 scale.

    Args:
        a (Path): First image.
        b (Path): Second image.

    Returns:
        float: RMS difference in grey levels; ``inf`` when the image shapes differ.
    """
    x, y = image.imread(a), image.imread(b)
    if x.shape != y.shape:
        return float("inf")
    return float(np.sqrt(np.mean((x.astype(float) - y.astype(float)) ** 2)) * rgb_scale)


def load_config() -> dict[str, dict]:
    """Read the per-example output table from ``tests/examples.json``.

    Returns:
        dict[str, dict]: Example name to its configuration; keys starting with an underscore are dropped.
    """
    raw = json.loads((ROOT / "tests" / "examples.json").read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def check_example(name: str, cfg: dict, pixel_tol: float, strict: bool, keep: bool) -> tuple[bool, float]:
    """Run one quick path into a temporary directory and compare its outputs with the committed ones.

    Args:
        name (str): Example folder name.
        cfg (dict): Entry of ``tests/examples.json`` for this example (its ``outputs`` map).
        pixel_tol (float): Maximum PNG pixel RMS difference that passes, in grey levels.
        strict (bool): Fail on PDF and JSON differences even when PNGs differ within the tolerance.
        keep (bool): Keep the temporary output directory for inspection.

    Returns:
        tuple[bool, float]: Whether the example passed, and the wall-clock time of the quick path in seconds.
    """
    out = Path(tempfile.mkdtemp(prefix=f"orchid-{name}-"))
    env = dict(os.environ, PYTHON=sys.executable, MPLBACKEND="Agg")
    t0 = time.time()
    r = subprocess.run(["sh", str(ROOT / name / "run_quick.sh"), str(out)], env=env, capture_output=True, text=True, check=False)
    dt = time.time() - t0
    if r.returncode:
        print(f"FAIL {name}: run_quick.sh exited {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return False, dt
    ok, png_exact, soft = True, True, []
    for f, ref in cfg["outputs"].items():
        got, want = out / f, ROOT / name / ref
        if not got.exists():
            print(f"FAIL {name}: {f} not written")
            ok = False
            continue
        if sha(got) == sha(want):
            print(f"  ok  {name}/{f}: identical (sha256 {sha(got)[:12]})")
            continue
        if f.endswith(".png") and pixel_tol > 0:
            d = rms(got, want)
            passed = d <= pixel_tol
            png_exact = False
            print(f"  {'ok ' if passed else 'FAIL'} {name}/{f}: bytes differ, pixel RMS {d:.3f} (tol {pixel_tol})")
            ok = ok and passed
        else:
            soft.append(f)
    for f in soft:
        if png_exact or strict:
            print(f"FAIL {name}/{f}: differs from {cfg['outputs'][f]}")
            ok = False
        else:
            print(f"  warn {name}/{f}: differs (PNG rasterisation differs on this platform; not failed without --strict)")
    print(f"{name}: quick path {dt:.0f} s")
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return ok, dt


def main(argv: list[str] | None = None) -> int:
    """Check the quick paths named on the command line (all examples by default).

    Args:
        argv (list[str] | None): Command-line arguments; None reads ``sys.argv``.

    Returns:
        int: 0 when every example regenerates its outputs, 1 otherwise.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("examples", nargs="*")
    ap.add_argument("--pixel-tol", type=float, default=0.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    names = a.examples or list(cfg)
    failed = [name for name in names if not check_example(name, cfg[name], a.pixel_tol, a.strict, a.keep)[0]]
    print("FAILED: " + ", ".join(sorted(set(failed))) if failed else "all quick paths regenerate identically")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
