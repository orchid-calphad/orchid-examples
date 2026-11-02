"""Read the example's inputs file from the shipped .json.gz when the .json is absent (BL-031)."""

from __future__ import annotations

import gzip
from pathlib import Path


def read_inputs(path: str | Path) -> tuple[bytes, str]:
    """Bytes of an inputs file, read from ``path`` or, if that is missing, from ``path + ".gz"`` (the shipped form).

    Args:
        path (str | Path): The inputs file as named on the command line.

    Returns:
        ``(data, name)``. The name drops ".gz" and the bytes are decompressed, so a hash of them matches the
        provenance recorded with an uncompressed copy.
    """
    path = Path(path)
    gz = path.with_name(path.name + ".gz")
    source = gz if not path.exists() and gz.exists() else path
    raw = source.read_bytes()
    if source.suffix == ".gz":
        return gzip.decompress(raw), source.name[:-3]
    return raw, source.name
