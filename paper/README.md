# Paper figures and tables

Scripts that turn the stored experiment results in `raw_results/` into the
figures and LaTeX tables of the paper.

```bash
make paper-assets          # from the repo root: everything -> assets/
```

or, inside `paper/`:

```bash
python build_paper_assets.py               # all figures + tables
python build_paper_assets.py --skip-tables # figures only
python datasets.py                          # show which run each dataset uses
```

Output goes to `assets/` (untracked): `assets/figures/*.pdf` and
`assets/table_*.tex`, the same layout as the paper repo. The location is set
once, in `datasets.py` (`ASSETS_DIR`).

## Stored results

`raw_results/` holds the 8 runs used in the paper (one `shapg_comparison_*`
and one `markovian_initial_*` run per dataset, made with the configs in `../configs/`).
Only the files the scripts read are tracked: `config.yaml`, `kpi_forward.json`,
`kpi_backward.json`, `timing.json`, `markovian_metadata.json`,
`shapley_values.csv`, and one transition graph.

`datasets.py` picks, per dataset, the newest run matching
`<prefix>_<dataset>_retrain*` (regression datasets prefer `_adj_r2_` runs).
To pin a run, set it in `SHAPG_OVERRIDES` / `MARKOVIAN_OVERRIDES`.

## Script → paper asset

| Script | Output (in `assets/`) | Reads |
|---|---|---|
| `generate_table_small.py`, `generate_table_big.py` | `table_results_small.tex` (Housing, H1N1), `table_results_big.tex` (Crime, MIC) | `shapg_comparison_*` |
| `replot_kpi.py` | `figures/<dataset>_kpi_{forward,backward}.pdf` | `shapg_comparison_*` (uses `../benchmark/plotting.py`) |
| `plot_importance_hist.py` | `figures/{housing,h1n1}_importance_hist.pdf` | `shapg_comparison_*/shapley_values.csv` |
| `generate_table_stochastic.py` | `table_stochastic.tex` | `markovian_initial_*` |
| `generate_table_stochastic_time.py` | `table_stochastic_time.tex` | `markovian_initial_*/timing.json` |
| `generate_table_absorbing.py` | `table_absorbing_housing.tex` | `markovian_initial_housing_*`, retrains LightGBM on `../data/housing_price.csv` |
| `plot_pareto_absorbing.py` | `figures/pareto_absorbing_housing.pdf` | `markovian_initial_housing_*`, `../data/housing_price.csv` |
| `plot_transition_example.py` | `figures/transition_example.pdf` | nothing (toy example in the script) |
| `build_paper_assets.py` | runs all of the above; copies the Housing transition graph | `markovian_initial_housing_*` |
| `exact_propagation.py` (run separately) | `exact_propagation_results.json` | `../data/housing_price.csv` |

`labels.py` holds the method names used in the tables.
