"""Generate table_results_big.tex (Crime, MIC) from the shapg_comparison runs."""

import json
import pathlib
import sys

import yaml

RESULTS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(RESULTS_DIR))

from labels import METHOD_SPEC, display

from datasets import ASSETS_DIR, INFO, SHAPG_DIRS, BIG, metric_label, shapg_path, threshold_key

# Dataset configurations — sourced from datasets.py so paths live in one place.
DATASETS = [
    {
        "dir": shapg_path(k),
        "label": INFO[k]["label"],
        "n_features": INFO[k]["n_features"],
        "metric_label": metric_label(k, SHAPG_DIRS[k], full=True),
        "threshold_key": threshold_key(k),
    }
    for k in BIG
]



def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def extract_method_data(ds_cfg):
    """Extract all metrics for one dataset, returning a list of row dicts."""
    d = RESULTS_DIR / ds_cfg["dir"]
    fwd = load_json(d / "kpi_forward.json")
    bwd = load_json(d / "kpi_backward.json")
    timing = load_json(d / "timing.json")
    markov_meta = load_json(d / "markovian_metadata.json")

    full_metric = fwd["_meta"]["full_metric"]

    rows = []
    for data_key, group in METHOD_SPEC:
        # Resolve threshold key per dataset
        key = ds_cfg["threshold_key"] if data_key == "__threshold__" else data_key
        latex_name = display(key)

        # Forward KPI
        if key not in fwd:
            continue
        fwd_entry = fwd[key]
        s_fwd = fwd_entry["slope"]
        metrics = fwd_entry["metrics"]
        peak = max(metrics)
        step = metrics.index(peak)  # index 0 = 0 features, so step = index
        pct_full = peak / full_metric * 100 if full_metric != 0 else 0

        # Backward KPI (oracle methods don't have it)
        if key in bwd:
            s_bwd = bwd[key]["slope"]
        else:
            s_bwd = None

        # Timing
        if key in timing:
            time_s = timing[key]
        elif "time" in fwd_entry:
            time_s = fwd_entry["time"]
        else:
            time_s = None

        # Markovian stopping point (absorbing state size)
        if key in markov_meta:
            n_stop = markov_meta[key]["stopping_point"]
        else:
            n_stop = None

        rows.append({
            "key": key,
            "latex_name": latex_name,
            "group": group,
            "s_fwd": s_fwd,
            "peak": peak,
            "step": step,
            "pct_full": pct_full,
            "s_bwd": s_bwd,
            "time": time_s,
            "n_stop": n_stop,
        })

    return rows, full_metric


def find_best_indices(rows):
    """Find indices of best non-oracle values for each metric. Returns dict of metric -> set of indices."""
    non_oracle = [(i, r) for i, r in enumerate(rows) if r["group"] != "oracle"]
    best = {}

    # S_fwd: higher is better
    vals = [(i, r["s_fwd"]) for i, r in non_oracle]
    max_val = max(v for _, v in vals)
    best["s_fwd"] = {i for i, v in vals if abs(v - max_val) < 1e-6}

    # Peak: higher is better
    vals = [(i, r["peak"]) for i, r in non_oracle]
    max_val = max(v for _, v in vals)
    best["peak"] = {i for i, v in vals if abs(v - max_val) < 1e-6}

    # %full: higher is better
    vals = [(i, r["pct_full"]) for i, r in non_oracle]
    max_val = max(v for _, v in vals)
    best["pct_full"] = {i for i, v in vals if abs(v - max_val) < 0.05}

    # S_bwd: higher is better (skip None)
    vals = [(i, r["s_bwd"]) for i, r in non_oracle if r["s_bwd"] is not None]
    if vals:
        max_val = max(v for _, v in vals)
        best["s_bwd"] = {i for i, v in vals if abs(v - max_val) < 1e-6}
    else:
        best["s_bwd"] = set()

    # Time: lower is better (skip None)
    vals = [(i, r["time"]) for i, r in non_oracle if r["time"] is not None]
    if vals:
        min_val = min(v for _, v in vals)
        best["time"] = {i for i, v in vals if abs(v - min_val) < 1e-6}
    else:
        best["time"] = set()

    return best


def fmt(value, precision, bold=False, thousands_sep=False):
    """Format a numeric value, optionally bold."""
    if value is None:
        return "---"
    if thousands_sep and value >= 1000:
        integer_part = int(value)
        decimal_part = value - integer_part
        # Format with \, thousands separator
        s = f"{integer_part:,}".replace(",", r"\,")
        s += f"{decimal_part:.1f}"[1:]  # append .X
    else:
        s = f"{value:.{precision}f}"
    if bold:
        s = r"\textbf{" + s + "}"
    return s


def generate_table():
    all_dataset_rows = []
    all_dataset_meta = []

    for ds_cfg in DATASETS:
        rows, full_metric = extract_method_data(ds_cfg)
        best = find_best_indices(rows)
        all_dataset_rows.append(rows)
        all_dataset_meta.append({
            "cfg": ds_cfg,
            "full_metric": full_metric,
            "best": best,
        })

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Consolidated results for Crime and MIC datasets. Best (except greedy best method) value per metric is in \textbf{bold}.}"
    )
    lines.append(r"\label{tab:results_new}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l ccccccc ccccccc}")
    lines.append(r"\toprule")

    # Dataset header row
    headers = []
    for meta in all_dataset_meta:
        cfg = meta["cfg"]
        fm = meta["full_metric"]
        headers.append(
            r"\multicolumn{7}{c}{\textbf{"
            + cfg["label"]
            + r"} ($n{=}"
            + str(cfg["n_features"])
            + r"$,\; $"
            + cfg["metric_label"]
            + r"{=}"
            + f"{fm:.3f}"
            + r"$)}"
        )
    lines.append("& " + "\n& ".join(headers) + r" \\")
    lines.append(r"\cmidrule(lr){2-8} \cmidrule(lr){9-15}")

    # Column header row
    col_hdr = (
        r"$S^{\mathrm{fwd}}$ & Peak & Step & \% full & $S^{\mathrm{bwd}}$ & $n_{\mathrm{stop}}$ & Time\,(s)"
    )
    lines.append(r"\textbf{Method}")
    lines.append(f"  & {col_hdr}")
    lines.append(f"  & {col_hdr} " + r"\\")
    lines.append(r"\midrule")

    # Data rows
    n_methods = len(all_dataset_rows[0])
    prev_group = None

    for i in range(n_methods):
        group = all_dataset_rows[0][i]["group"]
        if prev_group is not None and group != prev_group:
            lines.append(r"\cmidrule{1-15}")
        prev_group = group

        latex_name = all_dataset_rows[0][i]["latex_name"]
        cells = [latex_name]

        for ds_idx in range(len(DATASETS)):
            row = all_dataset_rows[ds_idx][i]
            best = all_dataset_meta[ds_idx]["best"]

            cells.append(fmt(row["s_fwd"], 3, bold=i in best["s_fwd"]))
            cells.append(fmt(row["peak"], 3, bold=i in best["peak"]))
            cells.append(str(row["step"]))
            cells.append(fmt(row["pct_full"], 1, bold=i in best["pct_full"]))
            cells.append(fmt(row["s_bwd"], 3, bold=i in best["s_bwd"]))
            cells.append(str(row["n_stop"]) if row["n_stop"] is not None else "---")
            cells.append(fmt(row["time"], 1, bold=i in best["time"], thousands_sep=True))

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    table = generate_table()
    print(table)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ASSETS_DIR / "table_results_big.tex"
    with open(output_path, "w") as f:
        f.write(table + "\n")
    print(f"\nWritten to {output_path}")
