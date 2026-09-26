#!/bin/bash
# Slurm array job for the v2 CrabNet rerun on BYU Research Computing (see README.md).
# Submit from $RERUN_DIR so that logs/ resolves there, for example:
#   cd ~/crabnet_rerun && sbatch --array=0-19 ~/matsci-opt-benchmarks/scripts/crabnet_hyperparameter/byu_rc/rerun.sbatch
# Command-line flags override the lines below, e.g. --gpus=l40s:1, --time=24:00:00,
# --qos=standby, --export=ALL,DESIGN=smoke.

#SBATCH --job-name=crabnet-v2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gpus=l40s:1
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out

set -eo pipefail

export RERUN_DIR="${RERUN_DIR:-$SLURM_SUBMIT_DIR}"
export MATMINER_DATA="$RERUN_DIR/matminer_data"
DESIGN="${DESIGN:-decision}"
REPO="${REPO:-$HOME/matsci-opt-benchmarks}"
ENV_NAME="${ENV_NAME:-crabnet-v2}"

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

module purge
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# CrabNet writes each fitted network (models/trained_models) and learning-curve data
# (figures/lc_data) to the working directory, so run from node-local scratch, which
# is removed when the job ends
WORKDIR="/tmp/$SLURM_JOB_ID"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT TERM
cd "$WORKDIR"

echo "task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname), design $DESIGN"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python "$REPO/scripts/crabnet_hyperparameter/byu_rc/rerun.py" run --design "$DESIGN"
