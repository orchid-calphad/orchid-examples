"""Metric blocks, bootstrap intervals and evidence clustering shared by the Cu-Zr scoring scripts.

Split out of score.py so that the equilibrium scoring and the grid scoring use the same metric code.
"""

from __future__ import annotations

import re

import numpy as np
from phase_metrics import phase_presence_metrics

nboot = 2000  # bootstrap resamples of every interval
seed = 0  # seed of the numpy generator that drives all bootstraps


def metric_block(truth: list[list[str]], pred: list[list[str]], rng: np.random.Generator) -> dict | None:
    """Metrics of one subset of points, with point-bootstrap 95% intervals.

    Args:
        truth (list[list[str]]): Reference assemblage per point.
        pred (list[list[str]]): Predicted assemblage per point, aligned with truth.
        rng (np.random.Generator): Generator that draws the bootstrap resamples (advanced in place).

    Returns:
        dict | None: Block with the point count, weighted metrics, macro metrics, exact-assemblage accuracy, the
        "ci95" percentile intervals (2.5 to 97.5) and the per-phase table; None when there are no points.
    """
    if not truth:
        return None
    m = phase_presence_metrics(truth, pred)
    exact = np.array([set(t) == set(q) for t, q in zip(truth, pred, strict=True)], float)
    n = len(truth)
    boots = {k: [] for k in ("recall", "precision", "f1", "accuracy", "exact")}
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        mb = phase_presence_metrics([truth[i] for i in idx], [pred[i] for i in idx])
        for k in ("recall", "precision", "f1", "accuracy"):
            boots[k].append(mb["weighted"][k])
        boots["exact"].append(exact[idx].mean())
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    return {
        "n_points": n,
        "weighted": m["weighted"],
        "macro_ground_truth_phases": m["macro_ground_truth_phases"],
        "exact_assemblage_accuracy": float(exact.mean()),
        "ci95": ci,
        "per_phase": m["per_phase"],
    }


def source_key(p: dict) -> str:
    """Cluster key of a literature point's original source, from the reference string.

    Keys are '<surname> <year>' so that '1953, C.E. Lundin, ...' and '1953, Lundin CE, ...' agree, and 'Liu 2017,
    doi:...' rows with or without a measured_in field agree. A reference that is only a DOI keys by that DOI. Years
    are taken as printed.

    Args:
        p (dict): Evidence point with a "provenance" mapping that holds the "reference" string.

    Returns:
        str: Cluster key of the original source.
    """
    ref = (p.get("provenance", {}).get("reference") or "").strip()
    if ref.lower().startswith("doi:"):
        return ref.split()[0].lower().rstrip(",")
    yr = re.search(r"(18|19|20)\d\d", ref)
    names = re.findall(r"[A-Z][a-z]{2,}", ref)
    return f"{names[0].lower() if names else ref[:20]} {yr.group(0) if yr else ''}".strip()


def cluster_ci(
    truth: list[list[str]], pred: list[list[str]], clusters: list[str], rng: np.random.Generator, nboot: int = nboot
) -> dict:
    """Percentile 95% intervals of weighted recall and exact accuracy with whole clusters resampled.

    Args:
        truth (list[list[str]]): Reference assemblage per point.
        pred (list[list[str]]): Predicted assemblage per point, aligned with truth.
        clusters (list[str]): Cluster key per point (equilibrium id or source key); points sharing a key resample together.
        rng (np.random.Generator): Generator that draws the resamples (advanced in place).
        nboot (int): Number of cluster resamples.

    Returns:
        dict: "n_clusters", and "recall" and "exact" as [low, high] intervals.
    """
    keys = sorted(set(clusters))
    idx_of = {k: [i for i, c in enumerate(clusters) if c == k] for k in keys}
    rec, ex = [], []
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = [i for k in pick for i in idx_of[keys[k]]]
        rec.append(phase_presence_metrics([truth[i] for i in idx], [pred[i] for i in idx])["weighted"]["recall"])
        ex.append(np.mean([set(truth[i]) == set(pred[i]) for i in idx]))
    return {
        "n_clusters": len(keys),
        "recall": [float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))],
        "exact": [float(np.percentile(ex, 2.5)), float(np.percentile(ex, 97.5))],
    }
