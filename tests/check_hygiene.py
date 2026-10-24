"""Repository hygiene: no absolute local paths, no file > 50 MB, total tracked size < 100 MB, PDFs only as generated
figures, no publisher/handbook material by file type, valid CITATION.cff / .zenodo.json basics.
  python tests/check_hygiene.py      (checks the files git tracks, or all files when not in a git checkout)"""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    files = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, check=True).stdout.split()
except Exception:
    files = []
if not files:
    files = [os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT) if "/.git" not in d + "/" for f in fs]
bad = []
pat = re.compile(r"(/Users/[A-Za-z]|/home/[a-z]|/private/tmp/|C:\\\\Users|~/projects/|\.venvs/)")
total = 0
for f in files:
    p = os.path.join(ROOT, f)
    if not os.path.isfile(p):
        continue
    n = os.path.getsize(p); total += n
    if n > 50 * 2**20:
        bad.append(f"{f}: {n / 2**20:.1f} MB > 50 MB")
    if f.lower().endswith((".pdf",)) and "/figures/" not in "/" + f:
        bad.append(f"{f}: PDF outside a figures/ folder")
    if f.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".djvu", ".epub")):
        bad.append(f"{f}: image/document type not expected")
    if f.endswith((".py", ".sh", ".md", ".json", ".yml", ".yaml", ".txt", ".csv", ".cff", ".ipynb", ".tdb", ".TDB")) and f != "tests/check_hygiene.py":
        for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
            if pat.search(line):
                bad.append(f"{f}:{i}: local absolute path: {line.strip()[:120]}")
if total > 100 * 2**20:
    bad.append(f"repository {total / 2**20:.1f} MB > 100 MB")
z = json.load(open(os.path.join(ROOT, ".zenodo.json")))
if "grants" in z:
    bad.append(".zenodo.json has 'grants'")
if re.search(r"^funding:", open(os.path.join(ROOT, "CITATION.cff")).read(), re.M):
    bad.append("CITATION.cff has a 'funding' key")
print(f"{len(files)} files, {total / 2**20:.1f} MB")
print("\n".join(bad) if bad else "hygiene ok")
sys.exit(1 if bad else 0)
