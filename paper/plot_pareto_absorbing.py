"""Pareto scatter of the Housing absorbing states (Top-k, S0=empty, M=50):
x = |S*| (size), y = v(S*) (adjusted R^2), marker area ~ sampling frequency,
Pareto-optimal states highlighted and connected (quality vs. size of the
absorbing states).

Writes assets/figures/pareto_absorbing_housing.pdf.
"""

import ast
import json
import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

try:
    import gnuplot_style as gp

    gp.use("all")
except ImportError:
    pass

HERE = pathlib.Path(__file__).resolve().parent
from datasets import FIGURES_DIR, markovian_path

DATA_CSV = HERE.parent / "data" / "housing_price.csv"
METHOD = "markovian_top_k_3_empty"

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
    meta = json.load(open(HERE / markovian_path("housing") / "markovian_metadata.json"))[METHOD]
    parse = lambda s: tuple(sorted(ast.literal_eval(s) if isinstance(s, str) else s))
    cnt = Counter(parse(p[-1]) for p in meta["scenario_paths"])
    pts = [(len(fs), v(list(map(int, fs))), freq) for fs, freq in cnt.items()]

    # Pareto front: no other point with size<=s and v>=vv (strictly better somewhere)
    def dominated(s, vv):
        return any(s2 <= s and v2 >= vv and (s2 < s or v2 > vv) for s2, v2, _ in pts)

    pareto = sorted([(s, vv, f) for s, vv, f in pts if not dominated(s, vv)])
    dom = [(s, vv, f) for s, vv, f in pts if dominated(s, vv)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(full, color="blue", ls="-.", lw=1.3, alpha=0.7,
               label=f"Full model ({full:.3f})")
    area = lambda f: 12 * f  # marker area strictly proportional to sampling frequency
    # dominated states
    ax.scatter([s for s, _, _ in dom], [vv for _, vv, _ in dom],
               s=[area(f) for _, _, f in dom], c="#9aa7b5",
               edgecolors="black", linewidths=0.5, alpha=0.8, label="absorbing state")
    # pareto states + connecting step line
    ax.plot([s for s, _, _ in pareto], [vv for _, vv, _ in pareto],
            color="#FF8C00", ls="--", lw=1.5, zorder=2)
    ax.scatter([s for s, _, _ in pareto], [vv for _, vv, _ in pareto],
               s=[area(f) for _, _, f in pareto], c="#FF8C00",
               edgecolors="black", linewidths=0.8, zorder=3, label="Pareto-optimal")

    # annotate most frequent and best
    mf = max(pts, key=lambda x: x[2])
    bv = max(pts, key=lambda x: x[1])
    ax.annotate(f"most frequent\n({mf[2]}/50)", (mf[0], mf[1]),
                textcoords="offset points", xytext=(8, -22), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("best $v(\\mathcal{S}^*)$", (bv[0], bv[1]),
                textcoords="offset points", xytext=(-66, 0), fontsize=9, va="center",
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel(r"Absorbing-state size $|\mathcal{S}^*|$", fontsize=13)
    ax.set_ylabel(r"$v(\mathcal{S}^*)$ (adjusted $R^2$)", fontsize=13)
    ax.set_xticks(range(min(s for s, _, _ in pts), max(s for s, _, _ in pts) + 1))
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    # note on marker size
    ax.text(0.02, 0.97, "marker area $\\propto$ sampling frequency",
            transform=ax.transAxes, fontsize=9, va="top", color="#555")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "pareto_absorbing_housing.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out.relative_to(HERE)}; Pareto front: "
          f"{[(s, round(vv,3)) for s,vv,_ in pareto]}")


if __name__ == "__main__":
    main()
