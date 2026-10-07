"""Re-plot the paper KPI figures from the shapg_comparison runs.

Uses the benchmark's plotting functions (``../benchmark/plotting.py``) and
plots all methods in each run except Random / Greedy Worst (benchmark
SKIP_METHODS) and CIS.

Writes ``assets/figures/<dataset>_kpi_forward.pdf`` and ``..._kpi_backward.pdf``.

Usage:
    python replot_kpi.py
"""

import json
import pathlib
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import the benchmark plotting machinery from ../benchmark.
sys.path.insert(0, str(HERE.parent))

from benchmark import labels as bench_labels  # noqa: E402
from benchmark.plotting import (  # noqa: E402
    plot_backward_kpi,
    plot_forward_kpi,
)

# Optional plotting style
try:
    import gnuplot_style as gp

    gp.use("all")
except ImportError:
    pass

from datasets import ALL, FIGURES_DIR, INFO, shapg_path  # noqa: E402

# Paper notation: improved ShapG variants without the dagger (matches tables).
# bench plotting reads these dicts at call time, so in-place updates apply.
bench_labels.DISPLAY_LABELS["shapg | rank_deletion_cosine"] = r"ShapG$_{\cos}$"
bench_labels.DISPLAY_LABELS["shapg | rank_deletion_kendalltau"] = r"ShapG$_{\tau}$"
bench_labels.DISPLAY_LABELS["shapg | rank_deletion_mutual_info"] = r"ShapG$_{MI}$"

# Paper notation: the method is renamed to MPE (Markovian Process Explainer).
# Rewrite the "Markovian" prefix on every label, keeping the sub/superscripts.
for _k, _v in list(bench_labels.DISPLAY_LABELS.items()):
    if _v.startswith("Markovian"):
        bench_labels.DISPLAY_LABELS[_k] = "MPE" + _v[len("Markovian"):]

# CIS is not in the paper; Random/Greedy Worst are skipped by SKIP_METHODS.
EXCLUDE = {"cis"}

FIG_DIR = FIGURES_DIR


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ds_key in ALL:
        src = HERE / shapg_path(ds_key)
        print(f"{INFO[ds_key]['label']}:")

        with open(src / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        fwd = load_json(src / "kpi_forward.json")
        bwd = load_json(src / "kpi_backward.json")
        markov_meta = load_json(src / "markovian_metadata.json")

        full_metric = fwd.pop("_meta", {}).get("full_metric")
        for key in EXCLUDE:
            fwd.pop(key, None)
            bwd.pop(key, None)

        metric_name = (
            "Adjusted $R^2$" if INFO[ds_key]["is_regression"] else "Accuracy"
        )

        # The benchmark functions write fixed filenames; render into a temp
        # dir, then move into figures/ with the dataset prefix.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            plot_forward_kpi(fwd, metric_name, full_metric, markov_meta, tmp, cfg)
            plot_backward_kpi(bwd, metric_name, tmp, cfg)
            for direction in ("forward", "backward"):
                dst = FIG_DIR / f"{ds_key}_kpi_{direction}.pdf"
                shutil.move(tmp / f"kpi_{direction}.pdf", dst)
                print(f"  wrote {dst.relative_to(HERE)}")


if __name__ == "__main__":
    main()
