"""BL-031: the MLIP example's default inputs path works on a fresh checkout and keeps the recorded provenance hash.

Only ``structures.json.gz`` is shipped, so the default ``structures.json`` must be read from it, and the hash of the
decompressed JSON must match the one in the cached calibration reference. Exit status 1 on failure.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
example = root / "mlip-formation-enthalpy"
sys.path.insert(0, str(example / "scripts" / "full"))
from inputs_io import read_inputs  # noqa: E402


def main() -> None:
    """Read the default inputs through ``read_inputs`` and compare their hash with the cached reference."""
    default = example / "inputs" / "structures.json"
    assert not default.exists(), "test assumes a fresh checkout (no gunzipped copy)"
    data, name = read_inputs(default)
    assert name == "structures.json", name
    assert len(json.loads(data)["items"]) > 0
    reference = json.loads((example / "cached" / "CALIBRATION_MLIP_reference.json").read_text(encoding="utf-8"))
    want = re.search(r"structures\.json ([0-9a-f]{64})", reference["pairs_release"]).group(1)
    got = hashlib.sha256(data).hexdigest()
    assert got == want, (got, want)
    print("ok: default inputs read from the shipped .gz; provenance hash", got[:12], "matches the cached reference")


if __name__ == "__main__":
    main()
