"""Locate ORCHID release files for the examples.

Every example pins the orchid-data commit it was built from (README of the example, and ``RELEASE`` in its scripts). A
release file is resolved, in this order, from

1. ``$ORCHID_DATA/<relpath>``: a local checkout of https://github.com/orchid-calphad/orchid-data or an unpacked Zenodo
   archive of the release. Check it out at the pinned commit (``git -C $ORCHID_DATA checkout <commit>``); a working tree
   with other changes gives other numbers.
2. the download cache ``$ORCHID_CACHE`` (default ``~/.cache/orchid-examples``), ``<cache>/<commit>/<relpath>``;
3. a download from ``$ORCHID_DATA_URL``, a template with ``{commit}`` and ``{path}``, by default the commit-pinned raw
   URL of the GitHub repository below. ``https://data.orchid-project.org/{path}`` serves the latest release (not
   commit-pinned) and works as well once the site is public.

Quick paths of the examples never call this module: they run from the cached outputs in each folder.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.request
from pathlib import Path

default_url = "https://raw.githubusercontent.com/orchid-calphad/orchid-data/{commit}/{path}"
site_url = "https://data.orchid-project.org/{path}"
download_timeout_s = 120  # seconds, per download
hash_chunk_bytes = 1 << 20  # 1 MiB read size when hashing


def cache_dir() -> Path:
    """Return the download cache directory.

    Returns:
        Path: ``$ORCHID_CACHE`` if set, else ``~/.cache/orchid-examples``.
    """
    return Path(os.environ.get("ORCHID_CACHE", Path.home() / ".cache" / "orchid-examples"))


def local_root() -> Path | None:
    """Return the local orchid-data checkout named by ``ORCHID_DATA``, if any.

    Returns:
        Path | None: The expanded ``$ORCHID_DATA`` directory, or None when the variable is unset or empty.
    """
    r = os.environ.get("ORCHID_DATA")
    return Path(r).expanduser() if r else None


def path(relpath: str, commit: str) -> Path:
    """Return the local path of a release file, downloading it into the cache when needed.

    Args:
        relpath (str): Posix path relative to the repository root, for example ``"espei/index.csv"``.
        commit (str): orchid-data commit the file is pinned to.

    Returns:
        Path: Existing local file.

    Raises:
        FileNotFoundError: ``ORCHID_DATA`` is set but the file is not in that checkout.
        RuntimeError: The file is not cached and the download failed.
    """
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
    url = os.environ.get("ORCHID_DATA_URL", default_url).format(commit=commit, path=relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=download_timeout_s) as r, tmp.open("wb") as f:
            f.write(r.read())
    except Exception as e:  # any network, HTTP or file error is reported with the remedy below
        raise RuntimeError(
            f"could not download {url}: {e}. Set ORCHID_DATA to a local orchid-data checkout at commit {commit}, "
            f"or ORCHID_DATA_URL to another mirror (template with {{commit}} and {{path}})."
        ) from e
    tmp.replace(p)
    return p


def listdir(reldir: str, commit: str, suffix: str = ".json") -> list[str]:
    """List the release files directly under a folder.

    Locally this is a directory listing; when downloading, the dataset list comes from ``espei/index.csv`` (ESPEI
    bundle) of the same commit, so only ``espei/`` folders can be listed remotely.

    Args:
        reldir (str): Release folder, for example ``"espei/experimental/CU-ZR"``.
        commit (str): orchid-data commit the folder is pinned to.
        suffix (str): File-name suffix to keep.

    Returns:
        list[str]: Sorted posix paths relative to the repository root.

    Raises:
        RuntimeError: A remote listing was requested for a folder outside ``espei/``.
    """
    reldir = reldir.strip("/")
    root = local_root()
    if root is not None:
        return sorted(f"{reldir}/{q.name}" for q in (root / reldir).glob("*" + suffix))
    if not reldir.startswith("espei/"):
        raise RuntimeError("remote listing is only available for espei/ folders; set ORCHID_DATA")
    rows = csv.DictReader(io.StringIO(path("espei/index.csv", commit).read_text()))
    sub = reldir[len("espei/") :]
    return sorted("espei/" + r["path"] for r in rows if r["path"].rsplit("/", 1)[0] == sub and r["path"].endswith(suffix))


def sha256(p: str | Path) -> str:
    """Return the sha256 hex digest of a file's bytes.

    Args:
        p (str | Path): File to hash.

    Returns:
        str: Lower-case hexadecimal digest.
    """
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(hash_chunk_bytes), b""):
            h.update(b)
    return h.hexdigest()
