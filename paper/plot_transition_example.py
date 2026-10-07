"""Draw the example transition subgraphs for the four transition rules.

Illustrates the worked example in the paper: state S = {x1,x2} with v(S)=0.5 and
four single-feature additions T1..T4.  For each rule (Linear, Top-k,
Top-p, Min-p) the transition probabilities are computed from the
example data with the parameters used in the experiments, then drawn as a
small star graph; discarded transitions are dashed gray.

Writes ``assets/figures/transition_example.pdf``.

Usage:
    python plot_transition_example.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from datasets import FIGURES_DIR

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = FIGURES_DIR

# ---- Example data -----------------------------------------------------------
V_S = 0.5
T_LABELS = [
    r"$T_1=\{x_1,x_2,x_3\}$" + "\n" + r"$v=0.55$",
    r"$T_2=\{x_1,x_2,x_5\}$" + "\n" + r"$v=0.70$",
    r"$T_3=\{x_1,x_2,x_6\}$" + "\n" + r"$v=0.75$",
    r"$T_4=\{x_1,x_2,x_8\}$" + "\n" + r"$v=0.75$",
]
DELTA = np.array([0.05, 0.20, 0.25, 0.25])

# ---- Rules (parameters as in the experiments) -------------------------------


def linear(d):
    return d / d.sum()


def boltzmann(d, tau):
    e = np.exp(d / tau)
    return e / e.sum()


def top_k(d, k=3, tau=1.0):
    sig = boltzmann(d, tau)
    keep = np.argsort(-sig, kind="stable")[:k]
    p = np.zeros_like(d)
    p[keep] = sig[keep] / sig[keep].sum()
    return p


def top_p(d, p=0.9, tau=0.1):
    sig = boltzmann(d, tau)
    order = np.argsort(-sig, kind="stable")
    cum = np.cumsum(sig[order])
    keep = order[: np.searchsorted(cum, p) + 1]
    out = np.zeros_like(d)
    out[keep] = sig[keep] / sig[keep].sum()
    return out


def min_p(d, p=0.5, tau=0.5):
    sig = boltzmann(d, tau)
    keep = sig >= p * sig.max()
    out = np.where(keep, sig, 0.0)
    return out / out.sum()


# The four transition rules (the Boltzmann distribution is only the building
# block of Top-p and Min-p, not a rule on its own).  Each entry is
# (title, final transition probabilities p_{S,T}, Boltzmann sigma_{S,T} or
# None).  For Top-p/Min-p the arrows show both sigma (used for filtering) and
# the renormalized p.
RULES = [
    ("Linear", linear(DELTA), None),
    (r"Top-$k$ ($k=3$, $\tau=1$)", top_k(DELTA, k=3, tau=1.0),
     boltzmann(DELTA, tau=1.0)),
    (r"Top-$p$ ($p=0.9$, $\tau=0.1$)", top_p(DELTA, p=0.9, tau=0.1),
     boltzmann(DELTA, tau=0.1)),
    (r"Min-$p$ ($p=0.5$, $\tau=0.5$)", min_p(DELTA, p=0.5, tau=0.5),
     boltzmann(DELTA, tau=0.5)),
]

# ---- Drawing ----------------------------------------------------------------

S_POS = (0.13, 0.5)
T_POS = [(0.85, y) for y in (0.875, 0.625, 0.375, 0.125)]


def draw_rule(ax, title, probs, sigmas=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    node_kw = dict(ha="center", va="center", fontsize=9, zorder=3)
    s_box = dict(boxstyle="round,pad=0.45", facecolor="lightblue",
                 edgecolor="black", linewidth=1.2)
    t_box = dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                 edgecolor="black", linewidth=1.0)

    ax.text(*S_POS, r"$S=\{x_1,x_2\}$" + "\n" + r"$v=0.5$",
            bbox=s_box, **node_kw)

    for i, ((x, y), label, p) in enumerate(zip(T_POS, T_LABELS, probs)):
        kept = p > 1e-12
        box = dict(t_box)
        if not kept:
            box.update(facecolor="whitesmoke", edgecolor="gray")
        ax.text(x, y, label, color="black" if kept else "gray",
                bbox=box, **node_kw)

        arrow = FancyArrowPatch(
            (S_POS[0] + 0.13, S_POS[1] + 0.55 * (y - S_POS[1])),
            (x - 0.17, y),
            arrowstyle="-|>", mutation_scale=14, zorder=2,
            color="black" if kept else "lightgray",
            linestyle="-" if kept else "--", linewidth=1.4 if kept else 1.0,
        )
        ax.add_patch(arrow)

        sub = f"S,T_{i + 1}"
        if sigmas is None:
            text = (f"$p_{{{sub}}}={p:.3f}$" if kept else f"$p_{{{sub}}}=0$")
        else:
            text = (rf"$\sigma_{{{sub}}}={sigmas[i]:.3f}$"
                    + "\n"
                    + (f"$p_{{{sub}}}={p:.3f}$" if kept else f"$p_{{{sub}}}=0$"))
        mx = (S_POS[0] + 0.13 + x - 0.17) / 2
        my = (S_POS[1] + 0.55 * (y - S_POS[1]) + y) / 2
        ax.text(mx, my + (0.075 if sigmas is not None else 0.04), text,
                ha="center", fontsize=8.5,
                color="black" if kept else "gray", zorder=4)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (title, probs, sigmas) in zip(axes.flat, RULES):
        draw_rule(ax, title, probs, sigmas)
    plt.tight_layout()

    out = FIG_DIR / "transition_example.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out.relative_to(HERE)}")
    for title, probs, _sigmas in RULES:
        print(f"  {title}: {np.round(probs, 3)}")


if __name__ == "__main__":
    main()
