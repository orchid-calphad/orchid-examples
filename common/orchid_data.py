"""Locate ORCHID release files for the examples.

Every example pins the orchid-data commit it was built from (README of the example, and ``RELEASE`` in its
scripts). A release file is resolved, in this order, from

1. ``$ORCHID_DATA/<relpath>``: a local checkout of https://github.com/orchid-calphad/orchid-data or an unpacked
   Zenodo archive of the release. Check it out at the pinned commit (``git -C $ORCHID_DATA checkout <commit>``);
   a working tree with other changes gives other numbers.
2. the download cache ``$ORCHID_CACHE`` (default ``~/.cache/orchid-examples``), ``<cache>/<commit>/<relpath>``;
3. a download from ``$ORCHID_DATA_URL``, a template with ``{commit}`` and ``{path}``, by default the
   commit-pinned raw URL of the GitHub repository below. ``https://data.orchid-project.org/{path}`` serves the
   latest release (not commit-pinned) and works as well once the site is public.

Quick paths of the examples never call this module: they run from the cached outputs in each folder.
"""
import csv
import hashlib
import io
import os
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://raw.githubusercontent.com/orchid-calphad/orchid-data/{commit}/{path}"
SITE_URL = "https://data.orchid-project.org/{path}"


def cache_dir():
    return Path(os.environ.get("ORCHID_CACHE", Path.home() / ".cache" / "orchid-examples"))


def local_root():
    r = os.environ.get("ORCHID_DATA")
    return Path(r).expanduser() if r else None


def path(relpath, commit):
    """Local path of release file ``relpath`` (posix, relative to the repository root) at ``commit``."""
    relpath = relpath.lstrip("/")
    root = local_root()
    if root is not None:
        p = root / relpath
        if not p.exists():
            raise FileNotFoundError(f"{p} (ORCHID_DATA={root}) does not exist")
        return p
    p = cache_dir() / commit / relpath
    if p.exists():
        return p
    url = os.environ.get("ORCHID_DATA_URL", DEFAULT_URL).format(commit=commit, path=relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"could not download {url}: {e}. Set ORCHID_DATA to a local orchid-data checkout at commit {commit}, "
            f"or ORCHID_DATA_URL to another mirror (template with {{commit}} and {{path}}).") from e
    tmp.replace(p)
    return p


def listdir(reldir, commit, suffix=".json"):
    """Release files directly under ``reldir`` (e.g. 'espei/experimental/CU-ZR'). Locally a directory listing; when
    downloading, the dataset list comes from espei/index.csv (ESPEI bundle) of the same commit."""
    reldir = reldir.strip("/")
    root = local_root()
    if root is not None:
        return sorted(f"{reldir}/{q.name}" for q in (root / reldir).glob("*" + suffix))
    if not reldir.startswith("espei/"):
        raise RuntimeError("remote listing is only available for espei/ folders; set ORCHID_DATA")
    rows = csv.DictReader(io.StringIO(path("espei/index.csv", commit).read_text()))
    sub = reldir[len("espei/"):]
    return sorted("espei/" + r["path"] for r in rows
                  if r["path"].rsplit("/", 1)[0] == sub and r["path"].endswith(suffix))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
