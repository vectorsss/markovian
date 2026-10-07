"""All plotting functions."""

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from shapG import MarkovianExplainer

from .explainers import node_to_feature
from .labels import SKIP_METHODS, display as _display

# Optional plotting style
try:
    import gnuplot_style as gp

    gp.use("all", cycle_mode="zip")
except ImportError:
    pass


def _get_plot_cfg(cfg):
    pcfg = cfg.get("plot", {})
    return pcfg.get("figsize", [12, 8]), pcfg.get("dpi", 300)


def plot_timing(explainer_results, output_dir, cfg):
    figsize, dpi = _get_plot_cfg(cfg)
    labels = list(explainer_results.keys())
    display_labels = [_display(l) for l in labels]
    times = [explainer_results[l]["time"] for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(display_labels, times)
    ax.set_xlabel("Time (s)")
    ax.set_title("Computation Time")
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 0.01 * max(times),
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "timing.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_backward_kpi(kpi_results, metric_name, output_dir, cfg):
    if not kpi_results:
        return
    figsize, dpi = _get_plot_cfg(cfg)
    fig, ax = plt.subplots(figsize=figsize)
    for method, data in kpi_results.items():
        if method in SKIP_METHODS:
            continue
        ax.plot(
            range(len(data["metrics"])),
            data["metrics"],
            label=f"{_display(method)} (S={data['slope']:.4f})",
        )
    ax.set_xlabel("Features Dropped", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_title("Backward KPI: Feature Dropping", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "kpi_backward.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def _plot_forward_on_ax(
    ax,
    forward_kpi_results,
    metric_name,
    full_metric,
    markovian_meta,
    xlim=None,
    ylim=None,
):
    """Plot forward KPI curves on a given axes."""
    reference_styles = {
        "Greedy Best": {
            "color": "black",
            "linestyle": "--",
            "marker": "x",
            "linewidth": 2,
            "alpha": 0.8,
        },
    }

    for method, data in forward_kpi_results.items():
        if method in SKIP_METHODS:
            continue
        metrics = data["metrics"]
        max_val = max(metrics)
        max_step = metrics.index(max_val)
        exceed_str = ""
        if full_metric is not None:
            exceed_step = next(
                (i for i, m in enumerate(metrics) if m >= full_metric), None
            )
            exceed_str = (
                f", ≥full@{exceed_step}" if exceed_step is not None else ", <full"
            )
        disp = _display(method)
        label = (
            f"{disp} (S={data['slope']:.4f}, max={max_val:.4f}@{max_step}{exceed_str})"
        )
        if method in reference_styles:
            ax.plot(
                range(len(metrics)), metrics, label=label, **reference_styles[method]
            )
        else:
            ax.plot(range(len(metrics)), metrics, label=label, alpha=0.6)

    if full_metric is not None:
        ax.axhline(
            y=full_metric,
            color="blue",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.7,
            label=f"Full model ({full_metric:.4f})",
        )

    # Markovian stopping points (hollow star), only if stop < curve length
    for method, meta in markovian_meta.items():
        if method in forward_kpi_results:
            m_metrics = forward_kpi_results[method]["metrics"]
            stop = meta["stopping_point"]
            if 0 < stop < len(m_metrics):
                ax.axvline(
                    x=stop,
                    color="red",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.5,
                )
                ax.plot(
                    stop,
                    m_metrics[stop],
                    marker="*",
                    markersize=15,
                    zorder=5,
                    markerfacecolor="none",
                    markeredgecolor="red",
                    markeredgewidth=1.5,
                )
                ax.annotate(
                    f"n={stop}",
                    (stop, m_metrics[stop]),
                    textcoords="offset points",
                    xytext=(8, -12),
                    fontsize=7,
                    color="red",
                )

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_xlabel("Number of Features Added", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)


def plot_forward_kpi(
    forward_kpi_results, metric_name, full_metric, markovian_meta, output_dir, cfg
):
    if not forward_kpi_results:
        return
    figsize, dpi = _get_plot_cfg(cfg)
    max_x = max(
        len(d["metrics"]) - 1
        for m, d in forward_kpi_results.items()
        if m not in SKIP_METHODS
    )

    fig, ax = plt.subplots(figsize=figsize)
    _plot_forward_on_ax(
        ax, forward_kpi_results, metric_name, full_metric, markovian_meta
    )
    ax.set_xticks(range(0, max_x + 1))
    ax.set_title("Forward KPI: Feature Addition", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "kpi_forward.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]


def plot_efficiency(fwd_data, timing_data, metric_name, full_metric, output_dir, cfg):
    """Scatter: x = computation time (s, log scale), y = 1 / peak performance.

    Each method gets a unique tab10 color and marker; labels are drawn near points.
    """
    _, dpi = _get_plot_cfg(cfg)
    fig, ax = plt.subplots(figsize=(8, 4))

    # Collect (time, 1/peak, display_label) for all methods
    points = []
    for method, data in fwd_data.items():
        if method in SKIP_METHODS:
            continue
        if method in timing_data:
            t = timing_data[method]
        elif "time" in data:
            t = data["time"]
        else:
            continue
        peak = max(data["metrics"])
        if peak <= 0:
            continue
        points.append((t, 1.0 / peak, _display(method)))

    if not points:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(points), 2)))

    for i, (x_val, y_val, label) in enumerate(points):
        ax.scatter(
            x_val,
            y_val,
            label=label,
            color=colors[i % len(colors)],
            marker=_MARKERS[i % len(_MARKERS)],
            edgecolors="black",
            s=100,
            zorder=4,
        )

    if full_metric is not None and full_metric > 0:
        ax.axhline(
            1.0 / full_metric,
            color="blue",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.7,
            label=f"1/full ({1.0/full_metric:.4f})",
        )

    # Set log scale last so no subsequent call can override it
    ax.set_xscale("log")
    ax.set_xlabel("Time Cost (s)", fontsize=11)
    ax.set_ylabel(f"$1/$peak {metric_name}", fontsize=11)
    ax.set_title("Efficiency: Time vs. Peak Performance", fontsize=13)
    ax.grid(alpha=0.3)

    # Legend outside below figure with box
    ncol = min(len(points) + 1, 5)
    ax.legend(
        title="Methods",
        fontsize=7,
        title_fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_shapley_values(explainer_results, columns, output_dir, cfg):
    figsize, dpi = _get_plot_cfg(cfg)
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(explainer_results.keys())
    n_methods = len(methods)
    bar_width = 0.8 / n_methods

    all_features = set()
    for res in explainer_results.values():
        for node in res["values"]:
            feat = node_to_feature(node, columns)
            if feat:
                all_features.add(feat)
    all_features = sorted(all_features)
    x = np.arange(len(all_features))

    for i, method in enumerate(methods):
        vals = []
        for feat in all_features:
            found = False
            for node, v in explainer_results[method]["values"].items():
                if node_to_feature(node, columns) == feat:
                    vals.append(v)
                    found = True
                    break
            if not found:
                vals.append(0.0)
        ax.bar(x + i * bar_width, vals, bar_width, label=_display(method))

    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Shapley Value")
    ax.set_title("Shapley Values Comparison")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "shapley_values.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Markovian-specific plots
# ──────────────────────────────────────────────────────────────────────────────


def find_markovian_explainers(explainer_results):
    """Find all (label, explainer) pairs that are live MarkovianExplainer instances."""
    return [
        (label, res["explainer"])
        for label, res in explainer_results.items()
        if isinstance(res.get("explainer"), MarkovianExplainer)
    ]


def plot_markovian_interaction_graph(explainer, label, columns, output_dir, cfg):
    """Visualize feature interaction graph from Markovian greedy process."""
    ig = explainer.build_interaction_graph()
    if ig.number_of_edges() == 0:
        print(f"    {label}: no interactions detected, skipping.")
        return

    label_map = {}
    for node in ig.nodes():
        name = node_to_feature(node, columns)
        label_map[node] = name if name else str(node)

    edges = list(ig.edges(data=True))
    weights = [d["weight"] for _, _, d in edges]
    max_weight = max(weights)
    edge_widths = [1.0 + 4.0 * (w / max_weight) for w in weights]

    figsize, dpi = _get_plot_cfg(cfg)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(ig, seed=42, k=2.0)

    nx.draw_networkx_nodes(ig, pos, node_size=600, node_color="lightblue", alpha=0.9)
    nx.draw_networkx_labels(ig, pos, labels=label_map, font_size=8)
    nx.draw_networkx_edges(
        ig, pos, width=edge_widths, alpha=0.6, edge_color="steelblue"
    )

    sorted_edges = sorted(edges, key=lambda e: e[2]["weight"], reverse=True)
    top_k = min(10, len(sorted_edges))
    edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in sorted_edges[:top_k]}
    nx.draw_networkx_edge_labels(ig, pos, edge_labels=edge_labels, font_size=7)

    safe_label = label.replace(" | ", "_").replace(" ", "_")
    plt.title(f"Interaction Graph: {_display(label)}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"interaction_graph_{safe_label}.pdf", dpi=dpi, bbox_inches="tight"
    )
    plt.close()

    print(f"    Top-{top_k} interactions ({label}):")
    for u, v, d in sorted_edges[:top_k]:
        print(
            f"      {label_map.get(u, u)} <-> {label_map.get(v, v)}: {d['weight']:.4f}"
        )


def plot_markovian_greedy_order(explainer, label, columns, output_dir, cfg):
    """Visualize greedy coalition formation order with marginal contributions."""
    greedy_order = explainer.get_greedy_order()
    shapley_values = explainer._shapley_values
    if not greedy_order:
        return

    names, marginals = [], []
    for node in greedy_order:
        name = node_to_feature(node, columns)
        names.append(name if name else str(node))
        marginals.append(shapley_values.get(node, 0.0))

    figsize, dpi = _get_plot_cfg(cfg)
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = plt.bar(
        range(len(names)), marginals, color=colors, edgecolor="black", alpha=0.8
    )

    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.xlabel("Feature (greedy addition order)")
    plt.ylabel("Marginal Contribution")
    plt.title(f"Greedy Order: {_display(label)}")
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars, marginals):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    safe_label = label.replace(" | ", "_").replace(" ", "_")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"greedy_order_{safe_label}.pdf", dpi=dpi, bbox_inches="tight"
    )
    plt.close()


def plot_markovian_process(explainer, label, columns, output_dir, cfg):
    """Visualize the Markovian coalition formation as a directed step graph."""
    history = explainer.get_process_history()
    if not history:
        return

    initial_entries = [e for e in history if e["action"] == "initial"]
    greedy_entries = [e for e in history if e["action"] != "initial"]

    dg = nx.DiGraph()

    if initial_entries:
        init_size = initial_entries[0]["coalition_size"]
        init_value = initial_entries[0]["value"]
        init_names = []
        for e in initial_entries:
            name = node_to_feature(e["player"], columns)
            init_names.append(name or str(e["player"]))
        init_label = f"Initial\n|S|={init_size}\nv={init_value:.4f}\n" + ", ".join(
            init_names
        )
        dg.add_node(0, label=init_label)
    else:
        dg.add_node(0, label="Step 0\n|S|=0\nv=0.0000")

    for i, entry in enumerate(greedy_entries):
        step_id = i + 1
        size = entry["coalition_size"]
        value = entry["value"]
        dg.add_node(step_id, label=f"Step {step_id}\n|S|={size}\nv={value:.4f}")

    edge_colors = []
    edge_labels = {}
    for i, entry in enumerate(greedy_entries):
        step_id = i + 1
        player = entry["player"]
        action = entry["action"]
        improvement = entry["improvement"]

        player_name = node_to_feature(player, columns) or str(player)
        prefix = "+" if action == "join" else "-"
        color = "green" if action == "join" else "red"

        dg.add_edge(i, step_id)
        edge_colors.append(color)
        edge_labels[(i, step_id)] = f"{prefix}{player_name}\n({improvement:+.4f})"

    n_nodes = len(dg.nodes())
    pos = {}
    if n_nodes <= 9:
        for i, node in enumerate(sorted(dg.nodes())):
            pos[node] = (i * 2.0, 0)
    else:
        row_width = 5
        for i, node in enumerate(sorted(dg.nodes())):
            row = i // row_width
            col = i % row_width
            if row % 2 == 1:
                col = row_width - 1 - col
            pos[node] = (col * 2.0, -row * 3.0)

    fig_width = max(12, min(n_nodes * 2.5, 24))
    fig_height = max(6, (n_nodes // 5 + 1) * 3.5) if n_nodes > 9 else 6
    figsize, dpi = _get_plot_cfg(cfg)
    plt.figure(figsize=(fig_width, fig_height))

    node_labels = nx.get_node_attributes(dg, "label")
    nx.draw_networkx_nodes(
        dg,
        pos,
        node_size=2000,
        node_color="lightyellow",
        edgecolors="black",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(dg, pos, labels=node_labels, font_size=7)
    nx.draw_networkx_edges(
        dg,
        pos,
        edge_color=edge_colors,
        width=2.0,
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edge_labels(
        dg, pos, edge_labels=edge_labels, font_size=6, font_color="black"
    )

    process_type = "Join-and-Leave" if explainer.allow_leave else "Monotone"
    legend_elements = [
        Line2D([0], [0], color="green", linewidth=2, label="Join (+feature)"),
        Line2D([0], [0], color="red", linewidth=2, label="Leave (-feature)"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.title(f"Markovian Process ({process_type}): {_display(label)}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    safe_label = label.replace(" | ", "_").replace(" ", "_")
    plt.savefig(
        output_dir / f"process_graph_{safe_label}.pdf", dpi=dpi, bbox_inches="tight"
    )
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Transition graph (Graphviz)
# ──────────────────────────────────────────────────────────────────────────────


def _compute_exact_transition_probs(
    S,
    v_func,
    candidate_ids,
    strategy,
    temperature,
    strategy_param,
    min_improvement,
    allow_leave,
):
    """Compute exact transition probabilities from state *S*.

    ``S`` and ``candidate_ids`` use the same identifier space (internal
    integer indices). ``v_func`` accepts those same identifiers.
    """
    v_S = v_func(S)

    join_deltas = {}
    for i in candidate_ids:
        if i not in S:
            delta = v_func(S | {i}) - v_S
            if delta >= min_improvement:
                join_deltas[i] = delta

    leave_deltas = {}
    if allow_leave:
        for j in S:
            delta = v_func(S - {j}) - v_S
            if delta > min_improvement:
                leave_deltas[j] = delta

    candidates = [
        (action, p, d)
        for action, dd in [("join", join_deltas), ("leave", leave_deltas)]
        for p, d in dd.items()
    ]
    if not candidates:
        return {}

    positive = [c for c in candidates if c[2] > 0]

    # Greedy: pick the single best (ties broken uniformly)
    if strategy == "greedy":
        pool = positive if positive else [c for c in candidates if c[2] >= 0]
        if not pool:
            return {}
        if positive:
            best = max(c[2] for c in positive)
            pool = [c for c in positive if abs(c[2] - best) < 1e-12]
        prob_each = 1.0 / len(pool)
        result = {}
        for action, player, _ in pool:
            T = frozenset(S | {player}) if action == "join" else frozenset(S - {player})
            result[T] = result.get(T, 0) + prob_each
        return result

    # Stochastic strategies: fall back to uniform over non-negative if no positive
    if not positive:
        non_neg = [c for c in candidates if c[2] >= 0]
        if not non_neg:
            return {}
        prob_each = 1.0 / len(non_neg)
        result = {}
        for action, player, _ in non_neg:
            T = frozenset(S | {player}) if action == "join" else frozenset(S - {player})
            result[T] = result.get(T, 0) + prob_each
        return result

    deltas = np.array([c[2] for c in positive])
    logits = deltas / temperature
    logits -= logits.max()
    exp_d = np.exp(logits)
    softmax_probs = exp_d / exp_d.sum()

    if strategy == "linear":
        probs = deltas / deltas.sum()
    elif strategy == "softmax":
        probs = softmax_probs
    elif strategy == "top_k":
        k = min(int(strategy_param or 1), len(positive))
        top_idx = np.argsort(softmax_probs)[-k:]
        probs = np.zeros(len(positive))
        probs[top_idx] = softmax_probs[top_idx]
        probs /= probs.sum()
    elif strategy == "top_p":
        p = strategy_param if strategy_param is not None else 0.9
        sorted_idx = np.argsort(softmax_probs)[::-1]
        cumsum = np.cumsum(softmax_probs[sorted_idx])
        cutoff = min(int(np.searchsorted(cumsum, p)) + 1, len(positive))
        top_idx = sorted_idx[:cutoff]
        probs = np.zeros(len(positive))
        probs[top_idx] = softmax_probs[top_idx]
        probs /= probs.sum()
    elif strategy == "min_p":
        p = strategy_param if strategy_param is not None else 0.1
        threshold = p * softmax_probs.max()
        mask = softmax_probs >= threshold
        probs = np.where(mask, softmax_probs, 0.0)
        probs /= probs.sum()
    else:
        probs = softmax_probs

    result = {}
    for (action, player, _), prob in zip(positive, probs):
        if prob < 1e-12:
            continue
        T = frozenset(S | {player}) if action == "join" else frozenset(S - {player})
        result[T] = result.get(T, 0) + prob
    return result


def plot_transition_graph(explainer, char_func, label, columns, output_dir, cfg):
    """Render the Markovian transition graph as a Graphviz PDF.

    Uses the live explainer's tracked transition data (``_transition_counts``,
    ``_scenario_paths``) so no re-simulation is needed.

    Configurable via ``cfg['plot']['transition_graph']``:
      - ``label_style``: ``"name"`` (default) or ``"index"`` — how players are
        rendered inside coalition nodes.
      - ``max_show``: max players to show before truncating with ``...``.
    """
    try:
        import graphviz
    except ImportError:
        print(f"    {label}: graphviz not installed, skipping transition graph.")
        return

    if explainer._transition_counts is None or explainer._scenario_paths is None:
        print(f"    {label}: no transition data available, skipping.")
        return

    transitions = explainer._transition_counts
    all_paths = explainer._scenario_paths
    n = explainer.n
    # explainer.player_ids holds the real player names (DataFrame columns),
    # but transitions/paths store frozensets of *internal integer indices*
    # (0..n-1), so we need to map indices → names when calling char_func.
    player_ids = list(explainer.player_ids)

    n_scenarios = getattr(explainer, "n_scenarios", 1)
    strategy = getattr(explainer, "strategy", "greedy")
    temperature = getattr(explainer, "temperature", 1.0)
    strategy_param = getattr(explainer, "strategy_param", None)
    min_improvement = getattr(explainer, "min_improvement", 0.0)
    allow_leave = getattr(explainer, "allow_leave", True)

    tg_cfg = cfg.get("plot", {}).get("transition_graph", {})
    label_style = tg_cfg.get("label_style", "name")
    max_show = tg_cfg.get("max_show", 5)

    full_set = frozenset(range(n))

    def _coal_label(S):
        """Format a coalition for node display."""
        if not S:
            return "∅"
        if S == full_set:
            return "𝒩"  # mathcal N
        if label_style == "index":
            names = [str(i) for i in sorted(S)]
        else:
            names = [str(player_ids[i]) for i in sorted(S)]
        if len(names) > max_show:
            return "{" + ", ".join(names[:max_show]) + ", ...}"
        return "{" + ", ".join(names) + "}"

    # v(S) wrapper: S contains internal indices, char_func expects player names.
    v_cache = {}

    def v_func(S):
        key = frozenset(S)
        if key not in v_cache:
            if not S:
                v_cache[key] = 0.0
            else:
                player_set = {player_ids[i] for i in S}
                v_cache[key] = float(char_func(player_set, None))
        return v_cache[key]

    # Absorbing states = last state of each scenario path
    absorbing = {path[-1] for path in all_paths if path}

    # All visited states
    visited_states = set()
    for path in all_paths:
        visited_states.update(path)

    strategy_label = strategy
    if strategy_param is not None:
        strategy_label += f" p={strategy_param}"
    strategy_label += f" τ={temperature}"

    # ── Build Graphviz Digraph ───────────────────────────────────────────────
    g = graphviz.Digraph(
        "transition",
        format="pdf",
        engine="dot",
        graph_attr={
            "rankdir": "LR",
            "ranksep": "1.5",
            "nodesep": "0.6",
            "fontname": "Helvetica",
            "label": f"Transition Graph (n={n}, {strategy_label}, M={n_scenarios})",
            "labelloc": "t",
            "fontsize": "16",
        },
        node_attr={
            "shape": "ellipse",
            "fontname": "Helvetica",
            "fontsize": "10",
            "margin": "0.15,0.08",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "8",
        },
    )

    # Invisible anchor chain: forces coalition sizes to flow left→right,
    # regardless of whether scenarios start from ∅ or 𝒩.
    sizes = sorted({len(s) for s in visited_states})
    for s in sizes:
        g.node(
            f"__anchor_{s}",
            label="",
            style="invis",
            shape="point",
            width="0.01",
            height="0.01",
        )
    for i in range(len(sizes) - 1):
        g.edge(
            f"__anchor_{sizes[i]}",
            f"__anchor_{sizes[i + 1]}",
            style="invis",
        )

    # Group same-size coalitions with their anchor in rank=same subgraphs
    layers = defaultdict(list)
    for S in visited_states:
        layers[len(S)].append(S)
    for size, states in sorted(layers.items()):
        with g.subgraph() as sub:
            sub.attr(rank="same")
            sub.node(f"__anchor_{size}")
            for S in states:
                sub.node(str(sorted(S)))

    # Outgoing totals for empirical probabilities
    outgoing_total = defaultdict(int)
    for S, targets in transitions.items():
        for cnt in targets.values():
            outgoing_total[S] += cnt

    # Nodes
    for S in visited_states:
        node_id = str(sorted(S))
        label_text = f"{_coal_label(S)}\nv={v_func(S):.3f}"

        if S in absorbing:
            g.node(
                node_id,
                label=label_text,
                style="filled",
                fillcolor="#FFF9E6",
                color="#FF8C00",
                penwidth="2.5",
                fontsize="10",
            )
        else:
            g.node(
                node_id,
                label=label_text,
                style="filled",
                fillcolor="white",
                color="black",
                penwidth="1.2",
                fontsize="10",
            )

    # Self-loops on absorbing states
    for S in absorbing:
        g.edge(
            str(sorted(S)),
            str(sorted(S)),
            label="",
            color="#FF8C00",
            penwidth="1.5",
            arrowsize="0.7",
        )

    # Exact transition probabilities. The explainer persists the exact
    # per-step distribution computed at sampling time (free, any n) — use it
    # whenever available; recompute only for states lacking persisted data
    # (e.g. monotone runs), which is feasible only for small n.
    exact_cache = dict(getattr(explainer, "_transition_probs", None) or {})
    if n <= 20:
        internal_ids = list(range(n))
        for S in visited_states:
            if S in absorbing or S in exact_cache:
                continue
            try:
                exact_cache[S] = _compute_exact_transition_probs(
                    S,
                    v_func,
                    internal_ids,
                    strategy,
                    temperature,
                    strategy_param,
                    min_improvement,
                    allow_leave,
                )
            except (ValueError, KeyError, OverflowError) as e:
                print(f"    exact probs error at {sorted(S)}: {e}")

    # Edges
    for S, targets in transitions.items():
        if S not in visited_states:
            continue
        src = str(sorted(S))
        exact_from_S = exact_cache.get(S, {})

        for T, count in targets.items():
            if T not in visited_states:
                continue
            dst = str(sorted(T))
            emp_prob = count / outgoing_total[S] if outgoing_total[S] > 0 else 0
            freq = count / max(n_scenarios, 1)
            exact_prob = exact_from_S.get(T)

            is_leave = len(T) < len(S)
            color = "#D62728" if is_leave else "#1F77B4"
            pw = max(0.8, 0.8 + 3.0 * freq)
            alpha_hex = format(int(max(80, min(255, 80 + 175 * freq))), "02x")

            if n_scenarios > 1:
                edge_label = f"{emp_prob:.0%} ({count}/{n_scenarios})"
                if exact_prob is not None:
                    edge_label += f"\n[p={exact_prob:.2f}]"
            else:
                edge_label = f"p={exact_prob:.2f}" if exact_prob is not None else ""

            g.edge(
                src,
                dst,
                label=edge_label,
                color=color + alpha_hex,
                fontcolor=color,
                penwidth=str(pw),
                arrowsize="0.8",
            )

    safe_label = label.replace(" | ", "_").replace(" ", "_")
    out_path = str(output_dir / f"transition_graph_{safe_label}")
    g.render(out_path, cleanup=True)
    print(f"    Saved transition graph: {out_path}.pdf")
