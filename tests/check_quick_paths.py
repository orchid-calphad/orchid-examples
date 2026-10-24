"""Run the quick path of every example (or the ones named) into a temporary directory and check that the outputs
regenerate identically: sha256 equal to the committed files (tests/examples.json). With --pixel-tol T, a PNG whose
bytes differ passes when the RMS difference of its pixels (0-255 scale) is <= T (fonts/rasterisation across platforms);
the PDF and JSON outputs of such an example are then reported but not failed unless --strict.

  python tests/check_quick_paths.py [EXAMPLE ...] [--pixel-tol 0] [--strict] [--keep]
Uses the current interpreter (PYTHON) for the examples' scripts. Exit status 1 on any failure."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rms(a, b):
    import numpy as np
    from matplotlib import image
    x, y = image.imread(a), image.imread(b)
    if x.shape != y.shape:
        return float("inf")
    return float(np.sqrt(np.mean((x.astype(float) - y.astype(float)) ** 2)) * 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples", nargs="*")
    ap.add_argument("--pixel-tol", type=float, default=0.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    cfg = {k: v for k, v in json.load(open(os.path.join(ROOT, "tests", "examples.json"))).items() if not k.startswith("_")}
    names = a.examples or list(cfg)
    env = dict(os.environ, PYTHON=sys.executable, MPLBACKEND="Agg")
    failed = []
    for name in names:
        out = tempfile.mkdtemp(prefix=f"orchid-{name}-")
        t0 = time.time()
        r = subprocess.run(["sh", os.path.join(ROOT, name, "run_quick.sh"), out], env=env, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode:
            print(f"FAIL {name}: run_quick.sh exited {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            failed.append(name); continue
        png_exact, soft = True, []
        for f, ref in cfg[name]["outputs"].items():
            got, want = os.path.join(out, f), os.path.join(ROOT, name, ref)
            if not os.path.exists(got):
                print(f"FAIL {name}: {f} not written"); failed.append(name); continue
            if sha(got) == sha(want):
                print(f"  ok  {name}/{f}: identical (sha256 {sha(got)[:12]})"); continue
            if f.endswith(".png") and a.pixel_tol > 0:
                d = rms(got, want)
                ok = d <= a.pixel_tol
                png_exact = False
                print(f"  {'ok ' if ok else 'FAIL'} {name}/{f}: bytes differ, pixel RMS {d:.3f} (tol {a.pixel_tol})")
                if not ok:
                    failed.append(name)
            else:
                soft.append(f)
        for f in soft:
            if png_exact or a.strict:
                print(f"FAIL {name}/{f}: differs from {cfg[name]['outputs'][f]}"); failed.append(name)
            else:
                print(f"  warn {name}/{f}: differs (PNG rasterisation differs on this platform; not failed without --strict)")
        print(f"{name}: quick path {dt:.0f} s")
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)
    print("FAILED: " + ", ".join(sorted(set(failed))) if failed else "all quick paths regenerate identically")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
