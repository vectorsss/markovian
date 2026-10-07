"""Generate table_stochastic_time.tex: run time in seconds of each stochastic MPE
variant (rule x initial state, M=50 scenarios) per dataset, read from the
markovian_initial timing.json files."""

import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).parent

from datasets import ASSETS_DIR, INFO, ALL, markovian_path

# Same strategy/init ordering and labels as generate_table_stochastic.py.
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


def load_timing(ds_key):
    path = RESULTS_DIR / markovian_path(ds_key) / "timing.json"
    with open(path) as f:
        return json.load(f)


def fmt(t):
    if t is None:
        return "---"
    return f"{t:.1f}"


def generate_table():
    timings = {k: load_timing(k) for k in ALL}

    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Computation time (in seconds) of the stochastic Markovian "
        r"process explainer ($M=50$ scenarios), by strategy and initial state. "
        r"Greedy is the single-scenario deterministic procedure, included for "
        r"reference.}"
    )
    lines.append(r"\label{tab:stochastic_time}")
    lines.append(r"\begin{tabular}{ll cccc}")
    lines.append(r"\toprule")

    header = r"\textbf{Strategy} & \textbf{Init}"
    for k in ALL:
        header += r" & \textbf{" + INFO[k]["label"] + r"}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\cmidrule(lr){3-6}")
    sub = "& & " + " & ".join([r"($n{=}" + str(INFO[k]["n_features"]) + r"$)" for k in ALL]) + r" \\"
    lines.append(sub)
    lines.append(r"\midrule")

    prev = None
    for strat_key, strat_label in STRATEGIES:
        for init_key, init_label in INITIALS:
            method = f"markovian_{strat_key}_{init_key}"
            if strat_key != prev:
                if prev is not None:
                    lines.append(r"\cmidrule{1-6}")
                cell = strat_label
                prev = strat_key
            else:
                cell = ""
            row = f"{cell} & {init_label}"
            for k in ALL:
                row += " & " + fmt(timings[k].get(method))
            row += r" \\"
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    table = generate_table()
    print(table)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / "table_stochastic_time.tex"
    with open(out, "w") as f:
        f.write(table + "\n")
    print(f"\nWritten to {out}")
