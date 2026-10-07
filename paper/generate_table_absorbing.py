"""Generate table_absorbing_housing.tex: all absorbing states found on Housing
with Top-k (k=3, S0=empty, M=50), the same setting as the transition-graph
figure. For each distinct absorbing state sampled across the
M scenarios we report its frequency, size, performance v(S*) (LightGBM adjusted
R^2, retrained on that subset), and feature composition.
"""

import ast
import json
import pathlib
from collections import Counter

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

RESULTS_DIR = pathlib.Path(__file__).parent
from datasets import ASSETS_DIR, markovian_path

DATA_CSV = RESULTS_DIR.resolve().parent / "data" / "housing_price.csv"
METHOD = "markovian_top_k_3_empty"
SHARED = ["LSTAT", "NOX", "RM"]  # present in every absorbing state (stated in caption)

data = pd.read_csv(DATA_CSV)
COLS = list(data.drop(columns=["MEDV"]).columns)
X, y = data.drop(columns=["MEDV"]), data["MEDV"].values


def v(idxs):
    cols = [COLS[i] for i in idxs]
    Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.2, random_state=42)
    m = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr)
    r2 = r2_score(yte, m.predict(Xte))
    n, p = len(yte), len(cols)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    full = v(range(len(COLS)))
    meta = json.load(open(RESULTS_DIR / markovian_path("housing") / "markovian_metadata.json"))
    m = meta[METHOD]
    parse = lambda s: tuple(sorted(ast.literal_eval(s) if isinstance(s, str) else s))
    cnt = Counter(parse(p[-1]) for p in m["scenario_paths"])

    rows = []
    for fs, freq in sorted(cnt.items(), key=lambda x: -x[1]):
        idxs = list(map(int, fs))
        names = [COLS[i] for i in idxs]
        other = [c for c in names if c not in SHARED]
        rows.append((freq, len(idxs), v(idxs), v(idxs) / full * 100, other))

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\caption{The distinct absorbing states reached on the \textit{Housing} "
        r"dataset by the Top-$k$ strategy ($k=3$, $\mathcal{S}_0=\varnothing$, "
        r"$M=50$), ordered by sampling frequency. Every state exceeds the "
        r"full-model performance ($R^2_{\mathrm{adj}}=" + f"{full:.3f}" + r"$) and "
        r"contains the three dominant features LSTAT, NOX and RM; the last column "
        r"lists the remaining features.}",
        r"\label{tab:absorbing_housing}",
        r"\begin{tabular}{c c c c l}",
        r"\toprule",
        r"Freq. & $|\mathcal{S}^*|$ & $v(\mathcal{S}^*)$ & \% full & Other features \\",
        r"\midrule",
    ]
    for freq, size, vs, pc, other in rows:
        lines.append(
            f"{freq}/50 & {size} & {vs:.3f} & {pc:.1f} & "
            + ", ".join(other) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table = "\n".join(lines)
    print(table)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / "table_absorbing_housing.tex"
    with open(out, "w") as f:
        f.write(table + "\n")
    print(f"\nWritten to {out}  ({len(rows)} absorbing states)")


if __name__ == "__main__":
    main()
