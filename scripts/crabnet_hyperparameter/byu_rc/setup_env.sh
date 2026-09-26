#!/bin/bash
# One-time setup on a BYU RC login node (compute nodes have no direct internet access):
# the conda environment, the v1 CSV from Zenodo and the matbench_expt_gap dataset.
#   bash setup_env.sh
# Uses REPO (default ~/matsci-opt-benchmarks), RERUN_DIR (default ~/crabnet_rerun) and
# ENV_NAME (default crabnet-v2).

set -eo pipefail

REPO="${REPO:-$HOME/matsci-opt-benchmarks}"
RERUN_DIR="${RERUN_DIR:-$HOME/crabnet_rerun}"
ENV_NAME="${ENV_NAME:-crabnet-v2}"
export MATMINER_DATA="$RERUN_DIR/matminer_data"
mkdir -p "$RERUN_DIR/logs" "$MATMINER_DATA"

module load miniforge3
eval "$(conda shell.bash hook)"
mamba create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

# The v1 versions: crabnet 2.0.8 (Python <= 3.10) with torch 1.13.1 built for CUDA
# 11.7. These exact builds are pip wheels; the CUDA runtime ships inside the torch
# wheel, so no cuda module is needed.
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install crabnet==2.0.8 "scikit-learn==1.3.2" "numpy<1.27" "pandas<2.1" \
    "pymatgen<2024.7" xtal2png joblib scipy
# matbench 0.6 pins scikit-learn 1.0.1, which has no Python 3.10 wheel
pip install --no-deps matbench==0.6 matminer
pip install --no-deps -e "$REPO"

python -c "from matsci_opt_benchmarks.crabnet_hyperparameter.utils.parameters import submitit_evaluate"
curl -sSL -o "$RERUN_DIR/sobol_regression.csv" \
    "https://zenodo.org/api/records/7694268/files/sobol_regression.csv/content"
python - <<'EOF'
from matbench.bench import MatbenchBenchmark

for task in MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"]).tasks:
    task.load()
EOF
ls -la "$RERUN_DIR" "$MATMINER_DATA"
