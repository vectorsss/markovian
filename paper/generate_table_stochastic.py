"""Generate table_stochastic.tex (stochastic MPE, all datasets) from the markovian_initial runs."""

import json
import pathlib
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent

from datasets import ASSETS_DIR, INFO, MARKOVIAN_DIRS, ALL, metric_label, markovian_path

# Dataset configurations — sourced from datasets.py so paths live in one place.
DATASETS = [
    {
        "dir": markovian_path(k),
        "label": INFO[k]["label"],
        "n_features": INFO[k]["n_features"],
        "metric_label": metric_label(k, MARKOVIAN_DIRS[k]),
    }
    for k in ALL
]

STRATEGIES = [
    ("greedy", r"Greedy"),
    ("linear", r"Linear"),
    ("top_p_09_t01", r"Top-$p$"),
    ("top_k_3", r"Top-$k$"),
    ("min_p_05_t05", r"Min-$p$"),
]

INITIALS = [
    ("empty", r"$\emptyset$"),
    ("rho", r"$\rho$"),
    ("full", r"$\mathcal{N}$"),
]


def generate_table():
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stochastic Markovian process results ($M=50$ scenarios). "
                 r"$v(\mathcal{S}^*)$: absorbing state performance (mean $\pm$ std); "
                 r"$|\mathcal{S}^*|$: absorbing state size ($[\min, \max]$ over the $M$ scenarios).}")
    lines.append(r"\label{tab:stochastic}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # 4 datasets x 2 metrics = 8 data columns + 2 label columns
    col_spec = "ll" + " cc" * 4
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Dataset headers
    header = r"\textbf{Strategy} & \textbf{Init}"
    for ds in DATASETS:
        header += r" & \multicolumn{2}{c}{\textbf{" + ds["label"] + r"} ($n{=}" + str(ds["n_features"]) + r"$)}"
    header += r" \\"
    lines.append(header)

    cmidrules = ""
    for i, ds in enumerate(DATASETS):
        col_start = 3 + i * 2
        col_end = col_start + 1
        cmidrules += r"\cmidrule(lr){" + str(col_start) + "-" + str(col_end) + "} "
    lines.append(cmidrules)

    # Sub-headers
    subheader = "& "
    for ds in DATASETS:
        subheader += r"& $v(\mathcal{S}^*)$ & $|\mathcal{S}^*|$ "
    subheader += r"\\"
    lines.append(subheader)
    lines.append(r"\midrule")

    # Load all data
    all_data = {}
    for ds in DATASETS:
        meta = json.load(open(RESULTS_DIR / ds["dir"] / "markovian_metadata.json"))
        fwd = json.load(open(RESULTS_DIR / ds["dir"] / "kpi_forward.json"))
        full_v = None
        for key, m in meta.items():
            fv = m.get("full_set_metric")
            if fv is not None:
                full_v = fv
                break
        # Greedy Best oracle
        gb_metrics = fwd.get("Greedy Best", {}).get("metrics", [])
        gb_peak = max(gb_metrics) if gb_metrics else None
        gb_step = gb_metrics.index(gb_peak) if gb_peak is not None else None
        all_data[ds["label"]] = {
            "meta": meta, "full_v": full_v,
            "gb_peak": gb_peak, "gb_step": gb_step,
        }

    # Full model row
    full_row = r"\multicolumn{2}{l}{\textit{Full model}} "
    for ds in DATASETS:
        fv = all_data[ds["label"]]["full_v"]
        full_row += f"& {fv:.3f} & {ds['n_features']} "
    full_row += r"\\"
    lines.append(full_row)

    # Greedy Best oracle row
    gb_row = r"\multicolumn{2}{l}{\textit{Greedy Best}} "
    for ds in DATASETS:
        gb_peak = all_data[ds["label"]]["gb_peak"]
        gb_step = all_data[ds["label"]]["gb_step"]
        if gb_peak is not None:
            gb_row += f"& {gb_peak:.3f} & {gb_step} "
        else:
            gb_row += "& --- & --- "
    gb_row += r"\\"
    lines.append(gb_row)
    lines.append(r"\midrule")

    # Data rows
    prev_strat = None
    for strat_key, strat_label in STRATEGIES:
        for init_key, init_label in INITIALS:
            method_key = f"markovian_{strat_key}_{init_key}"

            # Strategy label only on first row of group
            if strat_key != prev_strat:
                if prev_strat is not None:
                    lines.append(r"\cmidrule{1-" + str(2 + len(DATASETS) * 2) + "}")
                strat_cell = strat_label
                prev_strat = strat_key
            else:
                strat_cell = ""

            row = f"{strat_cell} & {init_label} "

            for ds in DATASETS:
                meta = all_data[ds["label"]]["meta"]
                if method_key not in meta:
                    row += "& --- & --- "
                    continue

                m = meta[method_key]
                metrics = m.get("absorbing_metrics", [])
                sizes = m.get("absorbing_state_sizes", [m["stopping_point"]])

                v_mean = np.mean(metrics)
                v_std = np.std(metrics)
                s_min, s_max = int(min(sizes)), int(max(sizes))

                if v_std < 1e-6:
                    v_str = f"{v_mean:.3f}"
                else:
                    v_str = f"{v_mean:.3f}{{\\tiny$\\pm${v_std:.3f}}}"

                # Integer [min, max] range avoids fractional coalition sizes,
                # which could mislead readers.
                if s_min == s_max:
                    s_str = f"{s_min}"
                else:
                    s_str = f"$[{s_min}, {s_max}]$"

                row += f"& {v_str} & {s_str} "

            row += r"\\"
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    table = generate_table()
    print(table)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ASSETS_DIR / "table_stochastic.tex"
    with open(output_path, "w") as f:
        f.write(table + "\n")
    print(f"\nWritten to {output_path}")
