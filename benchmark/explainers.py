"""Explainer execution, feature ranking, and Markovian metadata."""

import time

import lightgbm as lgb
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif

from shapG import (
    CISExplainer,
    LeverageScoreExplainer,
    MarkovianExplainer,
    ShapGExplainer,
)

from .shap_lib import SHAPLibKernelExplainer, SHAPLibSamplingExplainer

EXPLAINER_CLASSES = {
    "ShapGExplainer": ShapGExplainer,
    "CISExplainer": CISExplainer,
    "MarkovianExplainer": MarkovianExplainer,
    "LeverageScoreExplainer": LeverageScoreExplainer,
    # Official shap-library baselines (fixed model + masking, library defaults)
    "SHAPLibKernelExplainer": SHAPLibKernelExplainer,
    "SHAPLibSamplingExplainer": SHAPLibSamplingExplainer,
}


def compute_initial_coalition(X, y, G, threshold, model):
    """Select features with |correlation to target| >= threshold as initial coalition."""
    is_classifier = isinstance(model, lgb.LGBMClassifier)
    scores = {}

    if is_classifier:
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        X_numeric = X[numeric_cols].fillna(0)
        mi = mutual_info_classif(X_numeric, y, random_state=42)
        for col, s in zip(numeric_cols, mi):
            scores[col] = s
    else:
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                corr, _ = pearsonr(X[col].fillna(0), y)
                scores[col] = abs(corr)

    selected = [col for col, s in scores.items() if s >= threshold]
    if not selected:
        print(f"  No features exceed threshold {threshold}, starting from empty set.")
        return None

    # Map column names to graph node IDs
    node_ids = []
    graph_nodes = list(G.nodes())
    for col in selected:
        if col in graph_nodes:
            node_ids.append(col)
        else:
            idx = list(X.columns).index(col)
            if idx in graph_nodes:
                node_ids.append(idx)

    print(f"  Initial coalition via threshold={threshold}:")
    for col in selected:
        print(f"    {col}: score={scores[col]:.4f}")

    return node_ids if node_ids else None


def run_explainers(cfg, char_func, graphs, X, y, model, only=None):
    """Run every (explainer x graph) combination.

    Explainers with ``graph_dependent: false`` run once on the first graph.
    Markovian explainers with ``correlation_threshold`` get an initial coalition.
    If ``only`` is set, only run explainers whose names are in that list.
    """
    results = {}
    first_graph_name = next(iter(graphs))
    first_graph = graphs[first_graph_name]

    for exp_name, ecfg in cfg["explainers"].items():
        if only and exp_name not in only:
            continue

        cls = EXPLAINER_CLASSES[ecfg["class"]]
        params = dict(ecfg.get("params", {}))
        graph_dependent = ecfg.get("graph_dependent", True)
        verbose = params.pop("verbose", False)

        # shap-library baselines: fixed model + data, no characteristic function
        if getattr(cls, "requires_model_data", False):
            print(f"  Running {exp_name} ...")
            explainer = cls(
                model=model,
                X=X,
                y=y,
                split_cfg=cfg["model"]["train_test_split"],
                verbose=verbose,
                **params,
            )
            t0 = time.time()
            values = explainer.fit_explain()
            elapsed = time.time() - t0
            results[exp_name] = {
                "values": values,
                "time": elapsed,
                "explainer": explainer,
            }
            print(f"    done in {elapsed:.2f}s")
            continue

        # Handle Markovian correlation_threshold -> initial_coalition
        if cls is MarkovianExplainer and "correlation_threshold" in params:
            threshold = params.pop("correlation_threshold")
            G_for_init = first_graph
            initial_ids = compute_initial_coalition(X, y, G_for_init, threshold, model)
            if initial_ids:
                params["initial_coalition"] = initial_ids

        if graph_dependent:
            run_graphs = graphs.items()
        else:
            run_graphs = [(first_graph_name, first_graph)]

        for graph_name, G in run_graphs:
            label = f"{exp_name} | {graph_name}" if graph_dependent else exp_name
            print(f"  Running {label} ...")
            explainer = cls(
                characteristic_function=char_func, verbose=verbose, **params
            )

            t0 = time.time()
            values = explainer.fit_explain(G)
            elapsed = time.time() - t0

            results[label] = {"values": values, "time": elapsed, "explainer": explainer}
            print(f"    done in {elapsed:.2f}s")
    return results


def node_to_feature(node, columns):
    try:
        idx = int(node)
        if 0 <= idx < len(columns):
            return columns[idx]
    except (ValueError, TypeError):
        if node in columns:
            return str(node)
    return None


def to_feature_ranking(values, columns):
    sorted_vals = sorted(values.items(), key=lambda x: x[1], reverse=True)
    return [
        node_to_feature(n, columns)
        for n, _ in sorted_vals
        if node_to_feature(n, columns) is not None
    ]


def collect_markovian_metadata(explainer_results):
    """Extract stopping point from Markovian explainers."""
    meta = {}
    for label, res in explainer_results.items():
        exp = res["explainer"]
        if isinstance(exp, MarkovianExplainer):
            greedy_order = exp.get_greedy_order()
            entry = {
                "stopping_point": len(greedy_order),
                "n_players": exp.n,
                "greedy_order": [str(n) for n in greedy_order],
                "final_coalition": [str(n) for n in exp.get_final_coalition()],
            }
            # Absorbing state model performance
            if exp._absorbing_metrics is not None:
                metrics = exp._absorbing_metrics
                entry["absorbing_metric_mean"] = float(sum(metrics) / len(metrics))
                entry["absorbing_metric_min"] = float(min(metrics))
                entry["absorbing_metric_max"] = float(max(metrics))
                entry["absorbing_metrics"] = [float(m) for m in metrics]
            if exp._full_set_metric is not None:
                entry["full_set_metric"] = float(exp._full_set_metric)

            # Multi-scenario metadata
            if exp._scenario_finals is not None:
                scenario_sizes = [len(f) for f in exp._scenario_finals]
                unique_finals = len(set(exp._scenario_finals))
                entry["n_scenarios"] = exp.n_scenarios
                entry["strategy"] = exp.strategy
                entry["unique_absorbing_states"] = unique_finals
                entry["absorbing_state_sizes"] = scenario_sizes
                entry["unique_finals"] = [
                    sorted(str(n) for n in f) for f in set(exp._scenario_finals)
                ]
            # Transition data for graph visualization
            if exp._transition_counts is not None:
                # Convert frozenset keys to sorted string lists for JSON
                transitions_ser = {}
                for S, targets in exp._transition_counts.items():
                    src_key = str(sorted(str(n) for n in S))
                    transitions_ser[src_key] = {}
                    for T, cnt in targets.items():
                        dst_key = str(sorted(str(n) for n in T))
                        transitions_ser[src_key][dst_key] = cnt
                entry["transitions"] = transitions_ser
            if exp._scenario_paths is not None:
                entry["scenario_paths"] = [
                    [str(sorted(str(n) for n in s)) for s in path]
                    for path in exp._scenario_paths
                ]
            # Exact transition distributions captured at sampling time
            if getattr(exp, "_transition_probs", None):
                probs_ser = {}
                for S, targets in exp._transition_probs.items():
                    src_key = str(sorted(str(n) for n in S))
                    probs_ser[src_key] = {
                        str(sorted(str(n) for n in T)): float(p)
                        for T, p in targets.items()
                    }
                entry["transition_probs"] = probs_ser

            meta[label] = entry
    return meta
