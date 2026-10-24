# Copied verbatim (function body unchanged) from the PI's
#   crf_ternary/phase_metrics.py of the PhasePilot work (run phasepilot-fantasy128-20261006, crf-ternary),
#   sha256 640267ee3a836a5fbfb2abb5fcc85d367f44e22e3d3bd929f0f911575b1b6ead, copied 2026-10-24 for orchid-bench-pd (author: R. Arróyave's phasepilot-fantasy128 run).
"""Post-training, ground-truth-prevalence-weighted phase-presence metrics."""
import numpy as np


def phase_presence_metrics(truth, predicted):
    """Weight each phase by support / sum(supports), including coexistence.

    Labels are assemblages. This evaluates phase-model presence, not fractions
    or multiplicity. A phase absent from truth has zero prevalence weight but
    its false positives remain in the per-phase output. Undefined precision,
    recall, and F1 use zero, explicitly recorded below.
    """
    if len(truth) != len(predicted) or not len(truth):
        raise ValueError("Nonempty aligned truth and predictions required")
    truth, predicted = [set(v) for v in truth], [set(v) for v in predicted]
    if any(not v for v in truth):
        raise ValueError("Nonempty reference assemblages required")
    phases = sorted(set.union(*truth, *predicted))
    per_phase = {}
    for phase in phases:
        actual = np.array([phase in v for v in truth])
        guess = np.array([phase in v for v in predicted])
        tp, fn = int(np.sum(actual & guess)), int(np.sum(actual & ~guess))
        fp, tn = int(np.sum(~actual & guess)), int(np.sum(~actual & ~guess))
        support = tp + fn
        per_phase[phase] = {"support": support, "prevalence": support / len(truth),
                            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                            "accuracy": (tp + tn) / len(truth),
                            "precision": tp / (tp + fp) if tp + fp else 0.,
                            "recall": tp / support if support else 0.,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}
    total = sum(v["support"] for v in per_phase.values())
    for v in per_phase.values():
        v["weight"] = v["support"] / total
    keys = ("accuracy", "precision", "recall", "f1")
    return {"weighting": "ground_truth_phase_support_normalized_across_phases",
            "zero_division": 0, "phase_fraction_weighting": False,
            "weighted": {k: sum(v["weight"] * v[k] for v in per_phase.values()) for k in keys},
            "macro_ground_truth_phases": {k: float(np.mean([v[k] for v in per_phase.values() if v["support"]]))
                                          for k in keys},
            "per_phase": per_phase}
