"""Build paper-ready assets from raw_results/.

Runs the figure scripts, copies the selected transition graphs, and regenerates
the LaTeX tables. Everything goes to ``assets/`` (see ``datasets.ASSETS_DIR``):
``assets/figures/*.pdf`` and ``assets/table_*.tex``, the same layout as the
paper repo.

Usage:
    python build_paper_assets.py                     # figures + tables
    python build_paper_assets.py --skip-tables       # figures only

Add transition graphs by extending TRANSITIONS below.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from datasets import ASSETS_DIR, FIGURES_DIR  # noqa: E402
from datasets import MARKOVIAN_DIRS as DATASETS  # noqa: E402

# KPI figures are re-plotted by replot_kpi.py from the shapg_comparison runs,
# so they show the same methods as the tables.

# Transition graphs to ship: (dataset, method_key).
# Output name: ``transition_graph_<dataset>_<method>.pdf``.
TRANSITIONS = [
    ("housing", "markovian_top_k_3_empty"),
]

# Generators to re-run for the LaTeX tables.
GENERATORS = (
    "generate_table_small.py",
    "generate_table_big.py",
    "generate_table_stochastic.py",
    "generate_table_stochastic_time.py",
    "generate_table_absorbing.py",
)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    try:
        s_disp = str(src.relative_to(HERE))
    except ValueError:
        s_disp = str(src)
    try:
        d_disp = str(dst.relative_to(HERE))
    except ValueError:
        d_disp = str(dst)
    print(f"  {s_disp:70s}  ->  {d_disp}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--skip-tables", action="store_true",
                   help="Skip running the LaTeX table generators")
    args = p.parse_args()

    out = FIGURES_DIR
    raw = HERE / "raw_results"

    print(f"Source:  {raw}")
    print(f"Output:  {out}")
    print()

    n_kpi = n_trans = n_skip = 0

    # --- KPI plots (re-plotted from shapg_comparison JSON) ------------------
    print("KPI plots (replot_kpi.py):")
    r = subprocess.run([sys.executable, str(HERE / "replot_kpi.py")],
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL  replot_kpi.py (rc={r.returncode})")
        if r.stderr:
            print("    " + r.stderr.strip().splitlines()[-1])
        n_skip += 1
    else:
        for line in r.stdout.splitlines():
            if line.strip():
                print(f"  {line.strip()}")
        n_kpi = r.stdout.count("wrote ")
    print()

    # --- Standalone figure scripts ------------------------------------------
    for script in ("plot_importance_hist.py", "plot_transition_example.py",
                   "plot_pareto_absorbing.py"):
        print(f"Figures ({script}):")
        r = subprocess.run([sys.executable, str(HERE / script)],
                           cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL  {script} (rc={r.returncode})")
            if r.stderr:
                print("    " + r.stderr.strip().splitlines()[-1])
            n_skip += 1
        else:
            for line in r.stdout.splitlines():
                if line.strip():
                    print(f"  {line.strip()}")
        print()

    # --- Transition graphs -------------------------------------------------
    if TRANSITIONS:
        print("Transition graphs:")
        for dataset, method in TRANSITIONS:
            source = DATASETS.get(dataset)
            if source is None:
                print(f"  SKIP  unknown dataset '{dataset}'")
                n_skip += 1
                continue
            src = raw / source / f"transition_graph_{method}.pdf"
            if not src.exists():
                print(f"  SKIP  {dataset}/{method}")
                n_skip += 1
                continue
            copy(src, out / f"transition_graph_{dataset}_{method}.pdf")
            n_trans += 1
        print()

    # --- Tables ------------------------------------------------------------
    if args.skip_tables:
        print("Tables: skipped (--skip-tables)")
    else:
        print("Tables:")
        for gen in GENERATORS:
            r = subprocess.run([sys.executable, str(HERE / gen)],
                               cwd=HERE, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAIL  {gen}  (rc={r.returncode})")
                if r.stderr:
                    print("    " + r.stderr.strip().splitlines()[-1])
                n_skip += 1
                continue
            last = [l for l in r.stdout.splitlines() if l.strip()][-1]
            print(f"  OK    {gen}  ({last})")
    print()

    print(f"Done. kpi={n_kpi}  transitions={n_trans}  skipped={n_skip}")
    print(f"Paper-ready assets: {ASSETS_DIR}")
    if n_skip:
        sys.exit(1)


if __name__ == "__main__":
    main()
