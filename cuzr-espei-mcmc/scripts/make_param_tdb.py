"""Write tdb/CuZr-94Zen-LB_param.tdb and tdb/params.json for the SGTE Cu-Zr TDB with fitted coefficients as VV symbols.

The SGTE Cu-Zr TDB (Zeng et al. 1994; Hallstedt 2025, CC BY 4.0) has its 8 fitted coefficients replaced by FUNCTION
VV#### symbols (values unchanged). Copied from orchid-espei-demo scripts/make_start_tdb.py (v1 design); only the paths
changed. Run from the example root.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).absolute().parents[1]
os.chdir(root)
src = Path("tdb") / "CuZr-94Zen-LB.tdb"
# symbol, phase/parameter, old text, new text, value (J/mol, J/(mol*K) for the T-linear parts)
_fit_params = [
    ("VV0000", "L0(LIQUID) a", "-61685.53+11.29235*T", "VV0000+VV0001*T", -61685.53),
    ("VV0001", "L0(LIQUID) b", None, None, 11.29235),
    ("VV0002", "L1(LIQUID) a", "-8830.66+5.04565*T", "VV0002+VV0003*T", -8830.66),
    ("VV0003", "L1(LIQUID) b", None, None, 5.04565),
    ("VV0004", "G(CU51ZR14) a (J/mol-formula, 65 atoms)", "-843412.7", "+VV0004", -843412.7),
    ("VV0005", "G(CU10ZR7) a (J/mol-formula, 17 atoms)", "-241750", "+VV0005", -241750.0),
    ("VV0006", "G(CUZR_B2) a (J/mol-formula, 2 atoms)", "-20104.24-7.63196*T", "+VV0006-7.63196*T", -20104.24),
    ("VV0007", "G(CUZR2) a (J/mol-formula, 3 atoms)", "-43904.01+5.19051*T", "+VV0007+5.19051*T", -43904.01),
]


def write(path: str, values: dict[str, float], label: str) -> None:
    """Write the SGTE TDB with the VV symbols and their FUNCTION values in front of TEMP-LIM.

    Args:
        path (str): Output TDB path, relative to the example root.
        values (dict[str, float]): Value of each VV symbol, written with 6 decimals.
        label (str): Description of the model, written into the review-only header.
    """
    txt = src.read_text()
    for _sym, _desc, old, new, _ in _fit_params:
        if old is None:
            continue
        n = txt.count(old)
        assert n == 1, (old, n)
        txt = txt.replace(old, new)
    funcs = "".join(f"$ {s}: {d}\n FUNCTION {s:<9} 298.15 {values[s]:+.6f}; 6000 N !\n" for s, d, *_ in _fit_params)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    hdr = (
        f"$ REVIEW ONLY - NOT AN ASSESSMENT. {label}\n"
        f"$ Derived from {src.name} (sha256 {digest}) by orchid-examples cuzr-espei-mcmc/scripts/make_param_tdb.py.\n"
    )
    txt = txt.replace(" TEMP-LIM", hdr + funcs + " TEMP-LIM", 1)
    Path(path).write_text(txt)


if __name__ == "__main__":
    Path("tdb").mkdir(parents=True, exist_ok=True)
    start = {s: v for s, _, _, _, v in _fit_params}
    write("tdb/CuZr-94Zen-LB_param.tdb", start, "Starting model: SGTE values, fitted coefficients as VV symbols.")
    params_json = [{"symbol": s, "parameter": d, "start": v} for s, d, _, _, v in _fit_params]
    Path("tdb/params.json").write_text(json.dumps(params_json, indent=1))
