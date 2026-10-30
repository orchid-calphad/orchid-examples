"""Vendor the committed ORCHID calibration code and reference data, and import its statistics module.

Importing this module extracts the committed tools/ of orchid-project.org and the literature files of orchid-data into a
cache directory under the example (committed state only, never the working trees), puts tools/ on sys.path and imports
build_espei_bundle as ``bundle``. The revisions come from --site-rev and --data-rev, read before argparse runs.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # the vendored tools/ directory stays free of __pycache__
# git checkouts of orchid-calphad/orchid-project.org and orchid-calphad/orchid-data
site_repo = str(Path(os.environ.get("ORCHID_SITE_REPO", "orchid-project.org")).expanduser())
data_repo = str(Path(os.environ.get("ORCHID_DATA_REPO", "orchid-data")).expanduser())
root = Path(__file__).resolve().parents[2]  # example root


def _git_sha(repo: str, rev: str) -> str:
    """Full commit hash of a revision in a git checkout.

    Args:
        repo (str): Path of the git checkout.
        rev (str): Revision to resolve, for example HEAD or a commit hash.

    Returns:
        The full commit hash.
    """
    return subprocess.run(["git", "-C", repo, "rev-parse", rev], capture_output=True, text=True, check=True).stdout.strip()


def vendor_site(site_rev: str, data_rev: str) -> tuple[Path, str, str]:
    """Extract the committed calibration code and reference data into a cache directory.

    Committed state only (other agents' uncommitted edits are never used): orchid-project.org tools/ at site_rev
    (the DFT calibration code) + orchid-data literature/{dft_parity.json, orchid-literature.jsonl.gz, papers.json}
    at data_rev (the released experimental reference data and DFT pairs), laid out as <dir>/tools, <dir>/data/literature.

    Args:
        site_rev (str): orchid-project.org revision of the calibration code.
        data_rev (str): orchid-data revision of the reference data.

    Returns:
        Tuple (cache directory, orchid-project.org commit hash, orchid-data commit hash).
    """
    s1, s2 = _git_sha(site_repo, site_rev), _git_sha(data_repo, data_rev)
    d = root / ".cache" / f"orchid-{s1[:10]}-{s2[:10]}"
    if not (d / "ok").exists():
        (d / "data").mkdir(parents=True, exist_ok=True)
        ar = subprocess.run(["git", "-C", site_repo, "archive", s1, "tools"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=ar, check=True)
        ar = subprocess.run(
            [
                "git",
                "-C",
                data_repo,
                "archive",
                s2,
                "literature/dft_parity.json",
                "literature/orchid-literature.jsonl.gz",
                "literature/papers.json",
                "literature/index.json",
            ],
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(d / "data")], input=ar, check=True)
        (d / "ok").write_text(s1 + " " + s2, encoding="utf-8")
    return d, s1, s2


def _opt(name: str, default: str) -> str:
    """Value of a --name=value command-line option, read before argparse runs.

    Args:
        name (str): Option name including the leading dashes, for example --site-rev.
        default (str): Value when the option is absent.

    Returns:
        The option value, or default.
    """
    return next((x.split("=", 1)[1] for x in sys.argv if x.startswith(name + "=")), default)


site, site_sha, data_sha = vendor_site(_opt("--site-rev", "HEAD"), _opt("--data-rev", "HEAD"))
sys.path.insert(0, str(site / "tools"))
bundle = importlib.import_module("build_espei_bundle")  # imported from the vendored tools/ directory


def sha(p: str | Path) -> str:
    """SHA-256 of a file's bytes.

    Args:
        p (str | Path): File path.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(repo: str, *a: str) -> str | None:
    """Output of a git command in a checkout.

    Args:
        repo (str): Path of the git checkout.
        *a (str): Arguments after "git -C repo".

    Returns:
        The stripped standard output, or None if git fails.
    """
    try:
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=False).stdout.strip()
    except Exception:  # git missing, not a repository, or undecodable output: no git information
        return None


def committed_hull_tol() -> float:
    """Hull tolerance recorded in the committed DFT calibration of orchid-data.

    Returns:
        Hull tolerance in meV/atom, or 0.0 if the record is missing or malformed.
    """
    txt = git(data_repo, "show", "HEAD:espei/dft/CALIBRATION.json")
    try:
        m = json.loads(txt)["method"]
        return float(m.get("hull_tolerance_meV_per_atom", 0.0))
    except Exception:  # missing or malformed calibration record: no tolerance
        return 0.0
