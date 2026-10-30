"""Read the example's inputs file from the shipped .json.gz when the .json is absent (BL-031)."""
import gzip
import os


def read_inputs(path):
    """Bytes of an inputs file, read from ``path`` or, if that is missing, from ``path + ".gz"`` (the shipped form).

    Returns (bytes, name). ``name`` drops ".gz", and the bytes are decompressed, so a hash of them matches the
    provenance recorded with an uncompressed copy."""
    p = path if os.path.exists(path) or not os.path.exists(path + ".gz") else path + ".gz"
    raw = open(p, "rb").read()
    if p.endswith(".gz"):
        return gzip.decompress(raw), os.path.basename(p)[:-3]
    return raw, os.path.basename(p)
