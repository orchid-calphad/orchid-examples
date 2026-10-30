"""Paired bootstrap of metric differences between two scored models.

Same evidence points for both models; also prints the markdown tables
used in RESULTS.md. Usage: python compare.py [A.json B.json ...]  (default: GRACE vs SGTE and final vs SGTE)
Writes results/comparison.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parent))
from phase_metrics import phase_presence_metrics  # noqa: E402
from release import example_root  # noqa: E402

nb = 2000  # bootstrap resamples
seed = 0  # seed of the paired bootstrap generator


def _load(label: str) -> dict:
    """Load a scored result by its label.

    Args:
        label (str): Result label; the file is results/<label>.json.

    Returns:
        dict: Parsed result file.
    """
    return _load_path(example_root / "results" / f"{label}.json")


def _load_path(path: str | Path) -> dict:
    """Load a result JSON file.

    Args:
        path (str | Path): File to read.

    Returns:
        dict: Parsed result file.
    """
    return json.loads(Path(path).read_text())


def _weighted_recall(truth: list, pred: list) -> float:
    """Prevalence-weighted recall, the primary metric.

    Args:
        truth (list): Reference assemblage per point.
        pred (list): Predicted assemblage per point.

    Returns:
        float: Weighted recall in [0, 1].
    """
    return phase_presence_metrics(truth, pred)["weighted"]["recall"]


def paired(a: dict, b: dict, kind: str, mode: str = "physical") -> dict:
    """Paired bootstrap of the weighted-recall difference between two models on the same points.

    Args:
        a (dict): Result of the first model.
        b (dict): Result of the second model.
        kind (str): "field" or "boundary".
        mode (str): Class mode, "physical" or "strict".

    Returns:
        dict: Point count, the difference a - b in weighted recall, its 95% percentile interval and the one-sided
        bootstrap p-value.
    """
    pa = [p for p in a["points"] if p["kind"] == kind]
    pb = {p["id"]: p for p in b["points"] if p["kind"] == kind}
    truth = [p[f"truth_{mode}"] for p in pa]
    pred_a = [p[f"pred_{mode}"] for p in pa]
    pred_b = [pb[p["id"]][f"pred_{mode}"] for p in pa]
    rng = np.random.default_rng(seed)
    n = len(pa)
    d = []
    base = _weighted_recall(truth, pred_a) - _weighted_recall(truth, pred_b)
    for _ in range(nb):
        i = rng.integers(0, n, n)
        d.append(
            _weighted_recall([truth[k] for k in i], [pred_a[k] for k in i])
            - _weighted_recall([truth[k] for k in i], [pred_b[k] for k in i])
        )
    return {
        "n": n,
        "delta_weighted_recall": base,
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_boot_le0": float(np.mean(np.array(d) >= 0)) if base < 0 else float(np.mean(np.array(d) <= 0)),
    }


def _cell(weighted: dict, ci: dict, key: str) -> str:
    """Markdown cell of one metric: point value and its 95% interval.

    Args:
        weighted (dict): Weighted metrics of the block.
        ci (dict): Bootstrap intervals of the block.
        key (str): Metric name.

    Returns:
        str: "value [low, high]" text.
    """
    return f"{weighted[key]:.3f} [{ci[key][0]:.2f}, {ci[key][1]:.2f}]"


def row(lab: str, d: dict, kind: str, mode: str = "physical", sub: str = "all") -> str:
    """Markdown table row of one model.

    Args:
        lab (str): Model name for the row.
        d (dict): Result of the model.
        kind (str): "field" or "boundary".
        mode (str): Class mode, "physical" or "strict".
        sub (str): Point subset, "all" or "representable".

    Returns:
        str: Markdown table row.
    """
    b = d["metrics"][mode][kind][sub]
    w, c = b["weighted"], b["ci95"]
    return (
        f"| {lab} | {b['n_points']} | **{_cell(w, c, 'recall')}** | {_cell(w, c, 'precision')} | "
        f"{_cell(w, c, 'f1')} | {_cell(w, c, 'accuracy')} | "
        f"{b['exact_assemblage_accuracy']:.3f} [{c['exact'][0]:.2f}, {c['exact'][1]:.2f}] |"
    )


def _print_comparisons() -> dict:
    """Paired comparisons of the three model pairs, printed and returned.

    Returns:
        dict: Paired-bootstrap result per pair label and kind.
    """
    out = {}
    for a, b in (
        ("cuzr_grace", "cuzr_sgte_94Zen"),
        ("cuzr_phaseforgeplus_final", "cuzr_sgte_94Zen"),
        ("cuzr_phaseforgeplus_final", "cuzr_grace"),
    ):
        out[f"{a}-{b}"] = {k: paired(_load(a), _load(b), k) for k in ("field", "boundary")}
    return out


def _print_tables() -> None:
    """Print the markdown tables of every class mode, point subset and kind."""
    print("\n## tables")
    for mode in ("physical", "strict"):
        for sub in ("all", "representable"):
            for kind in ("field", "boundary"):
                print(
                    f"\n{mode} / {sub} / {kind}\n| model | n | weighted recall | weighted precision | weighted F1 | weighted accuracy | exact assemblage |\n|---|---|---|---|---|---|---|"
                )
                for lab, name in (
                    ("cuzr_grace", "GRACE-2L-OMAT MLIP-only"),
                    ("cuzr_sgte_94Zen", "SGTE CuZr-94Zen-LB"),
                    ("cuzr_phaseforgeplus_final", "PhaseForge+ final (in-sample)"),
                ):
                    print(row(name, _load(lab), kind, mode, sub))


def _print_panel() -> None:
    """Print the reference-panel table: one row per panel TDB and the SGTE reference."""
    print("\npanel")
    print("| TDB | field recall | field exact | boundary recall | boundary exact |\n|---|---|---|---|---|")
    files = sorted(str(p) for p in (example_root / "results" / "reference_panel").glob("*.json"))
    files.append(str(example_root / "results" / "cuzr_sgte_94Zen.json"))
    for f in files:
        d = _load_path(f)
        m = d["metrics"]["physical"]
        print(
            f"| {Path(d['tdb']).name} | {_recall_cell(m, 'field')} | {m['field']['all']['exact_assemblage_accuracy']:.3f} | "
            f"{_recall_cell(m, 'boundary')} | {m['boundary']['all']['exact_assemblage_accuracy']:.3f} |"
        )


def _recall_cell(m: dict, kind: str) -> str:
    """Panel-table cell with the weighted recall and its 95% interval of one kind.

    Args:
        m (dict): Physical-mode metric blocks of a result.
        kind (str): "field" or "boundary".

    Returns:
        str: "value [low, high]" text.
    """
    return f"{m[kind]['all']['weighted']['recall']:.3f} [{m[kind]['all']['ci95']['recall'][0]:.2f}, {m[kind]['all']['ci95']['recall'][1]:.2f}]"


def main() -> None:
    """Write results/comparison.json and print the comparisons and the markdown tables."""
    out = _print_comparisons()
    with (example_root / "results" / "comparison.json").open("w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(k, {kk: (round(vv["delta_weighted_recall"], 3), [round(x, 3) for x in vv["ci95"]]) for kk, vv in v.items()})
    _print_tables()
    _print_panel()


if __name__ == "__main__":
    main()
