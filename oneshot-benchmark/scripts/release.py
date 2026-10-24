"""Pinned ORCHID release for this example and path helpers."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "common"))
import orchid_data  # noqa: E402

RELEASE = "c3d9781c26f72baaad64dd2b6aa6126ba487659e"   # orchid-data commit the evidence was built from


def rel(p):
    """Repository-relative path for provenance fields (never an absolute local path)."""
    p = os.path.abspath(p)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else os.path.basename(p)
