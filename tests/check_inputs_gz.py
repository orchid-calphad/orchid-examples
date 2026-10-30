"""BL-031: the MLIP example's default inputs path works on a fresh checkout (only structures.json.gz is shipped), and the
recorded provenance hash is unchanged (hash of the decompressed JSON). Exit status 1 on failure."""
import hashlib, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "mlip-formation-enthalpy")
sys.path.insert(0, os.path.join(EX, "scripts", "full"))
from inputs_io import read_inputs  # noqa: E402

default = os.path.join(EX, "inputs", "structures.json")
assert not os.path.exists(default), "test assumes a fresh checkout (no gunzipped copy)"
data, name = read_inputs(default)
assert name == "structures.json", name
assert len(json.loads(data)["items"]) > 0
ref = json.load(open(os.path.join(EX, "cached", "CALIBRATION_MLIP_reference.json")))["pairs_release"]
want = re.search(r"structures\.json ([0-9a-f]{64})", ref).group(1)
got = hashlib.sha256(data).hexdigest()
assert got == want, (got, want)
print("ok: default inputs read from the shipped .gz; provenance hash", got[:12], "matches the cached reference")
