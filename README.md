# Markovian Process Explainer: experiment code

Code for the experiments in *Markovian process explainer: A new explainable
artificial intelligence method* (C. Zhao, E. Parilina, G. Zaccour).

The Markovian process explainer (MPE) itself is implemented in the
[shapG](https://pypi.org/project/shapG/) package (`shapG.MarkovianExplainer`).
This repository contains the benchmark that compares it with ShapG, KernelSHAP,
SamplingSHAP, and LeverageSHAP, and the stochastic MPE experiments.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The transition-graph figures also need the Graphviz `dot` binary
(e.g. `brew install graphviz`); without it they are skipped.

## Data

`benchmark/data.py` reads the processed datasets from `data/`. Crime and MIC are
included; Housing and H1N1 must be rebuilt first (`data/prepare_housing.py`,
`data/prepare_h1n1.py`). See `data/README.md` for sources, licenses, and the
processing steps.

| Dataset | File | Task | Features | Source |
|---|---|---|---|---|
| Housing | `housing_price.csv` | regression (`MEDV`) | 13 | Boston housing |
| H1N1 | `process_data.csv` | classification (`h1n1_vaccine`) | 35 | DrivenData, *Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines* |
| Crime | `crimedata_processing.csv` | regression (`ViolentCrimesPerPop`) | 120 | UCI, *Communities and Crime Unnormalized* (id 211) |
| MIC | `MIC_processing.csv` | classification (`LET_IS`, survival vs. death) | 111 | UCI, *Myocardial infarction complications* (id 579) |

## Running

```bash
bash run_all.sh            # all 8 configs
bash run_all.sh housing    # only the Housing configs
```

or a single config:

```bash
python run_benchmark.py configs/shapg_comparison_housing_retrain_adj_r2.yaml
```

Output goes to `results/<config>_<timestamp>/` (KPI JSON files, timings,
Shapley values, Markovian metadata, and figures).

| Config | Paper results |
|---|---|
| `shapg_comparison_{housing_retrain_adj_r2, h1n1_retrain, crimedata_retrain_adj_r2, mic_retrain}` | deterministic MPE vs. other methods: comparison tables, KPI figures, importance histograms |
| `markovian_initial_{housing_retrain_adj_r2, h1n1_retrain, crimedata_retrain_adj_r2, mic_retrain}` | stochastic MPE: results and timing tables, absorbing states, transition graph |

## Paper figures and tables

The results used in the paper are stored in `paper/raw_results/`. To rebuild
every figure and table of the paper from them:

```bash
make paper-assets    # -> paper/assets/figures/*.pdf and paper/assets/table_*.tex
```

See `paper/README.md` for which script makes which figure or table.

## Notes


Regression datasets use adjusted R², classification datasets use accuracy;
models are LightGBM, retrained on each feature subset.
`run_all.sh` sets `OMP_WAIT_POLICY=active` (as in the paper runs); with passive
OpenMP, LightGBM and ShapG run much slower.
