"""Exact probability-mass propagation for the stochastic Markovian explainer.

Computes, without Monte-Carlo sampling, the exact probability of reaching each
absorbing state under the four stochastic rules.

Every transition requires a strictly positive improvement, so v strictly
increases along any path => the reachable state graph is a DAG.
We expand states in increasing v order (heap), pushing probability mass
through the exact rule distributions (same semantics as
``MarkovianExplainer._select_action`` / ``_compute_exact_transition_probs``).
Mass arriving at a state with no positive candidate is absorbed.

Dataset: Housing (n=13, full lattice 2^13 = 8192 subsets -> bounded cost).
Start state: empty set. Rules and parameters exactly as in the paper:
Linear, Top-k (k=3, tau=1), Top-p (p=0.9, tau=0.1), Min-p (p=0.5, tau=0.5).

Outputs ``assets/exact_propagation_results.json`` and prints a summary.
"""

import heapq
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "housing_price.csv"

data = pd.read_csv(DATA)
X, y = data.drop(["MEDV"], axis=1), data["MEDV"]
FEATURES = list(X.columns)
N = len(FEATURES)

# ---- characteristic function (identical to the benchmark retrain mode) ------
_v_cache = {}


def v(S: frozenset) -> float:
    if S in _v_cache:
        return _v_cache[S]
    if not S:
        val = 0.0
    else:
        cols = [FEATURES[i] for i in sorted(S)]
        Xtr, Xte, ytr, yte = train_test_split(
            X[cols], y, test_size=0.2, random_state=42
        )
        m = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        m.fit(Xtr, ytr)
        r2 = r2_score(yte, m.predict(Xte))
        n_te, p = len(yte), len(S)
        val = r2 if n_te <= p + 1 else 1 - (1 - r2) * (n_te - 1) / (n_te - p - 1)
    _v_cache[S] = val
    return val


# ---- exact rule distributions (mirroring _select_action) --------------------


def rule_probs(deltas: np.ndarray, rule: str, tau: float, param):
    """deltas: positive improvements of the candidates. Returns probabilities."""
    if rule == "linear":
        return deltas / deltas.sum()
    logits = deltas / tau
    logits = logits - logits.max()
    sig = np.exp(logits)
    sig = sig / sig.sum()
    if rule == "top_k":
        k = min(int(param), len(deltas))
        keep = np.argsort(sig)[-k:]
        p = np.zeros_like(sig)
        p[keep] = sig[keep]
        return p / p.sum()
    if rule == "top_p":
        order = np.argsort(sig)[::-1]
        cum = np.cumsum(sig[order])
        cutoff = min(int(np.searchsorted(cum, param)) + 1, len(deltas))
        keep = order[:cutoff]
        p = np.zeros_like(sig)
        p[keep] = sig[keep]
        return p / p.sum()
    if rule == "min_p":
        mask = sig >= param * sig.max()
        p = np.where(mask, sig, 0.0)
        return p / p.sum()
    raise ValueError(rule)


def candidates(S: frozenset):
    """Positive-improvement single-feature transitions from S (join + leave)."""
    vS = v(S)
    out = []
    for i in range(N):
        if i not in S:
            d = v(S | {i}) - vS
            if d > 1e-12:
                out.append((S | {i}, i, d))
    for j in S:
        d = v(S - {j}) - vS
        if d > 1e-12:
            out.append((S - {j}, j, d))
    return out


# ---- DAG propagation ---------------------------------------------------------


def propagate(rule: str, tau: float, param, eps: float = 0.0):
    start = frozenset()
    pending = {start: 1.0}
    heap = [(v(start), 0, start)]
    discovered = {start}
    counter = 1

    absorbing = {}
    psi = np.zeros(N)
    expanded = 0
    pruned_mass = 0.0
    max_branch = 0
    t0 = time.time()

    while heap:
        _, _, S = heapq.heappop(heap)
        mass = pending.pop(S)
        cands = candidates(S)
        if not cands:
            absorbing[S] = absorbing.get(S, 0.0) + mass
            continue
        expanded += 1
        deltas = np.array([c[2] for c in cands])
        probs = rule_probs(deltas, rule, tau, param)
        max_branch = max(max_branch, int((probs > 1e-15).sum()))
        for (T, feat, d), p in zip(cands, probs):
            if p < 1e-15:
                continue
            m = mass * p
            if m < eps:
                pruned_mass += m
                continue
            psi[feat] += m * d
            if T in pending:
                pending[T] += m
            else:
                pending[T] = m
                if T not in discovered:
                    discovered.add(T)
                heapq.heappush(heap, (v(T), counter, T))
                counter += 1

    elapsed = time.time() - t0
    return {
        "rule": rule,
        "tau": tau,
        "param": param,
        "expanded_states": expanded,
        "n_absorbing": len(absorbing),
        "absorbing": sorted(
            (
                {
                    "features": [FEATURES[i] for i in sorted(S)],
                    "size": len(S),
                    "v": v(S),
                    "mass": m,
                }
                for S, m in absorbing.items()
            ),
            key=lambda a: -a["mass"],
        ),
        "total_mass": float(sum(absorbing.values())),
        "pruned_mass": pruned_mass,
        "max_branching": max_branch,
        "psi": {FEATURES[i]: float(psi[i]) for i in range(N) if psi[i] > 1e-12},
        "seconds": elapsed,
    }


RULES = [
    ("linear", 1.0, None),
    ("top_k", 1.0, 3),
    ("top_p", 0.1, 0.9),
    ("min_p", 0.5, 0.5),
]


def main():
    results = {}
    for rule, tau, param in RULES:
        print(f"=== {rule} (tau={tau}, param={param}) ===", flush=True)
        r = propagate(rule, tau, param)
        results[rule] = r
        ev_mean = sum(a["v"] * a["mass"] for a in r["absorbing"])
        es_mean = sum(a["size"] * a["mass"] for a in r["absorbing"])
        print(
            f"  expanded={r['expanded_states']}  absorbing={r['n_absorbing']}  "
            f"mass={r['total_mass']:.6f}  max_branch={r['max_branching']}  "
            f"E[v(S*)]={ev_mean:.4f}  E[|S*|]={es_mean:.2f}  "
            f"({r['seconds']:.0f}s, cache={len(_v_cache)})",
            flush=True,
        )
        for a in r["absorbing"][:5]:
            print(
                f"    mass={a['mass']:.4f}  v={a['v']:.4f}  |S*|={a['size']}  "
                f"{a['features']}",
                flush=True,
            )

    results["_meta"] = {
        "dataset": "housing",
        "n_features": N,
        "v_evaluations": len(_v_cache),
        "start": "empty",
    }
    from datasets import ASSETS_DIR

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / "exact_propagation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
