"""Plot feature-importance histograms of the deterministic Markovian explainer.

For Housing and H1N1, draws bars of the Shapley I scenario-values ``phi_i`` of
Markovian_empty over the features of its final model (6 for Housing, 9 for
H1N1), sorted by decreasing ``phi_i``.

Data: ``shapley_values.csv`` of the shapg_comparison runs.

Writes ``assets/figures/<dataset>_importance_hist.pdf``.

Usage:
    python plot_importance_hist.py
"""

import csv
import pathlib
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Optional plotting style
try:
    import gnuplot_style as gp

    gp.use("all")
except ImportError:
    pass

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import FIGURES_DIR, INFO, shapg_path  # noqa: E402

FIG_DIR = FIGURES_DIR

DET_METHOD = "markovian_empty"

DATASETS = ["housing", "h1n1"]


def load_values(csv_path):
    """shapley_values.csv -> {method: {feature: value}}."""
    out = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["method"]][row["node"]] = float(row["shapley_value"])
    return out


def plot_one(ds_key):
    det = load_values(HERE / shapg_path(ds_key) / "shapley_values.csv")[DET_METHOD]

    # Features of the final model, sorted by phi descending.
    feats = [k for k, v in sorted(det.items(), key=lambda x: -x[1]) if abs(v) > 1e-12]
    phi = [det[f] for f in feats]

    x = np.arange(len(feats))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, phi, 0.6, color="steelblue", edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(r"Feature importance $\phi_i$", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    out = FIG_DIR / f"{ds_key}_importance_hist.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out.relative_to(HERE)}  (sum phi={sum(phi):.4f})")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ds_key in DATASETS:
        print(f"{INFO[ds_key]['label']}:")
        plot_one(ds_key)


if __name__ == "__main__":
    main()
