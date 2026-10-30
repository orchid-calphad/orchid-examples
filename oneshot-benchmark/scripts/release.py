"""Pinned ORCHID release for this example, path helpers and file hashing."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

example_root = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(example_root.parent / "common"))
import orchid_data  # noqa: E402

__all__ = ["example_root", "file_sha256", "orchid_data", "rel", "release_commit"]

release_commit = "c3d9781c26f72baaad64dd2b6aa6126ba487659e"  # orchid-data commit the evidence was built from


def rel(p: str | Path) -> str:
    """Repository-relative path for provenance fields (never an absolute local path).

    Args:
        p (str | Path): File path, absolute or relative to the working directory.

    Returns:
        str: Path relative to the example folder when the file lies inside it, otherwise the bare file name.
    """
    full = Path(os.path.normpath(Path(p).absolute()))
    if full.is_relative_to(example_root):
        return str(full.relative_to(example_root))
    return full.name


def file_sha256(p: str | Path) -> str:
    """SHA-256 digest of a file's bytes, as recorded in the provenance of the outputs.

    Args:
        p (str | Path): File to hash.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
