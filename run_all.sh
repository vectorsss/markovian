#!/bin/bash
# Run all experiments of the paper (8 configs: 4 datasets x 2 experiment types).
#   shapg_comparison_*  : deterministic MPE vs. ShapG, KernelSHAP, SamplingSHAP, LeverageSHAP
#   markovian_initial_* : stochastic MPE (rules x initial states, M = 50 scenarios)
# Usage: bash run_all.sh            (all configs)
#        bash run_all.sh housing    (only configs whose name contains "housing")

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Settings used for the paper runs. Override by exporting beforehand.
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-active}"   # LightGBM is much slower with passive OpenMP
export RANK_DESC="${RANK_DESC:-1}"                    # ShapG edge pruning: descending (shapG default)
export MIXED_RANK_DESC="${MIXED_RANK_DESC:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILTER="${1:-}"

CONFIGS=(
    "configs/shapg_comparison_housing_retrain_adj_r2.yaml"
    "configs/shapg_comparison_h1n1_retrain.yaml"
    "configs/shapg_comparison_crimedata_retrain_adj_r2.yaml"
    "configs/shapg_comparison_mic_retrain.yaml"
    "configs/markovian_initial_housing_retrain_adj_r2.yaml"
    "configs/markovian_initial_h1n1_retrain.yaml"
    "configs/markovian_initial_crimedata_retrain_adj_r2.yaml"
    "configs/markovian_initial_mic_retrain.yaml"
)

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)
    if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
        continue
    fi
    output_dir="results/${name}_${TIMESTAMP}"
    log_dir="${output_dir}/logs"
    mkdir -p "$log_dir"

    echo "============================================================"
    echo "Running: $config"
    echo "Output:  $output_dir"
    echo "============================================================"

    python -u run_benchmark.py "$config" --output "$output_dir" \
        2>&1 | tee "${log_dir}/run.log"

    echo ""
    echo "Finished: $config"
    echo ""
done

echo "============================================================"
echo "All done. Results in: results/*_${TIMESTAMP}/"
echo "============================================================"
