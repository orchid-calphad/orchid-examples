"""Statistics of ORCHID's DFT-vs-experiment calibration (Dataset 4), used unchanged for the MLIP benchmark.

Copied from orchid-project.org tools/build_espei_bundle.py at commit 181ebd6001a41f93de539a288fbf96cb1f5bdab2 (MIT
licence, same copyright holder as this repository): median, quantile, stats_of, boot_ci_cluster. n_boot = 1000 as
there. Only the layout and names were restyled; the computation is unchanged. Robust sigma = 1.4826 x median absolute
deviation; the bootstrap seeds are derived from a label string, so every interval is reproducible exactly.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict

n_boot = 1000  # bootstrap resamples per interval, as in build_espei_bundle.py
_mad_to_sigma = 1.4826  # sigma of a normal distribution = 1.4826 x median absolute deviation


def median(xs: list[float]) -> float | None:
    """Median of a sample, with the mean of the two middle values for an even count.

    Args:
        xs (list[float]): Sample values, in any order.

    Returns:
        The median, or None for an empty sample.
    """
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(xs: list[float], q: float) -> float | None:
    """Linearly interpolated quantile of a sample.

    Args:
        xs (list[float]): Sample values, in any order.
        q (float): Quantile in [0, 1].

    Returns:
        The interpolated quantile, or None for an empty sample.
    """
    s = sorted(xs)
    if not s:
        return None
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def stats_of(res: list[float]) -> dict[str, float | int] | None:
    """Summary statistics of residuals (computed - experiment).

    Args:
        res (list[float]): Residuals, in kJ/mol-atom.

    Returns:
        Dict with n, bias_mean, bias_median, sigma_robust (kJ/mol-atom), q68_abs, q95_abs (kJ/mol-atom) and
        rmse (kJ/mol-atom); None for an empty sample.
    """
    n = len(res)
    if not n:
        return None
    med = median(res)
    mad = median([abs(x - med) for x in res])
    a = [abs(x) for x in res]
    return {
        "n": n,
        "bias_mean": sum(res) / n,
        "bias_median": med,
        "sigma_robust": _mad_to_sigma * mad,
        "q68_abs": quantile(a, 0.68),
        "q95_abs": quantile(a, 0.95),
        "rmse": math.sqrt(sum(x * x for x in res) / n),
    }


def boot_ci_cluster(ps: list[dict], label: str) -> dict[str, list[float]]:
    """System-cluster bootstrap of the residual statistics (audit A4 E8).

    Resamples chemical systems with replacement and pools their residuals, so that several records of one compound or
    system (CuZr: 12) are not counted as independent evidence.

    Args:
        ps (list[dict]): Pairs, each with a "system" key and a "res" key (residual, kJ/mol-atom).
        label (str): Seed label; the same label always gives the same intervals.

    Returns:
        Dict mapping each statistic (bias_median, sigma_robust, q68_abs, q95_abs, rmse) to its 95 % interval [low, high]
        in the statistic's units (kJ/mol-atom).
    """
    by = defaultdict(list)
    for p in ps:
        by[p["system"]].append(p["res"])
    systems = sorted(by)
    rng = random.Random(int(hashlib.sha256(("cluster|" + label).encode()).hexdigest()[:8], 16))
    acc = defaultdict(list)
    k = len(systems)
    for _ in range(n_boot):
        res = []
        for _ in range(k):
            res.extend(by[systems[rng.randrange(k)]])
        st = stats_of(res)
        for key in ("bias_median", "sigma_robust", "q68_abs", "q95_abs", "rmse"):
            acc[key].append(st[key])
    return {key: [round(quantile(v, 0.025), 3), round(quantile(v, 0.975), 3)] for key, v in sorted(acc.items())}
