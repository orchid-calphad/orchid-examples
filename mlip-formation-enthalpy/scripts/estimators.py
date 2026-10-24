"""Statistics of ORCHID's DFT-vs-experiment calibration (Dataset 4), used unchanged for the MLIP benchmark.

Copied verbatim (function bodies unchanged) from orchid-project.org tools/build_espei_bundle.py at commit
181ebd6001a41f93de539a288fbf96cb1f5bdab2 (MIT licence, same copyright holder as this repository): median,
quantile, stats_of, boot_ci_cluster. N_BOOT = 1000 as there. Robust sigma = 1.4826 x median absolute deviation;
the bootstrap seeds are derived from a label string, so every interval is reproducible exactly."""
import hashlib
import math
import random
from collections import defaultdict

N_BOOT = 1000


def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(xs, q):
    s = sorted(xs)
    if not s:
        return None
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def stats_of(res):
    n = len(res)
    if not n:
        return None
    med = median(res)
    mad = median([abs(x - med) for x in res])
    a = [abs(x) for x in res]
    return {"n": n, "bias_mean": sum(res) / n, "bias_median": med, "sigma_robust": 1.4826 * mad,
            "q68_abs": quantile(a, 0.68), "q95_abs": quantile(a, 0.95), "rmse": math.sqrt(sum(x * x for x in res) / n)}


def boot_ci_cluster(ps, label):
    """System-cluster bootstrap (audit A4 E8): resample chemical systems with replacement and pool their residuals,
    so that several records of one compound or system (CuZr: 12) are not counted as independent evidence."""
    by = defaultdict(list)
    for p in ps:
        by[p["system"]].append(p["res"])
    systems = sorted(by)
    rng = random.Random(int(hashlib.sha256(("cluster|" + label).encode()).hexdigest()[:8], 16))
    acc = defaultdict(list)
    k = len(systems)
    for _ in range(N_BOOT):
        res = []
        for _ in range(k):
            res.extend(by[systems[rng.randrange(k)]])
        st = stats_of(res)
        for key in ("bias_median", "sigma_robust", "q68_abs", "q95_abs", "rmse"):
            acc[key].append(st[key])
    return {key: [round(quantile(v, 0.025), 3), round(quantile(v, 0.975), 3)] for key, v in sorted(acc.items())}
