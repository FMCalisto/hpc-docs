#!/bin/bash
# Runs after every task of the sweep has succeeded (see submit.sh).
#SBATCH --job-name=kalman-collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "${ENV_DIR:-$HOME/envs/slurm-examples}/bin/activate"
python "$SLURM_SUBMIT_DIR/collect.py" "${RESULTS_DIR:-$HOME/results/kalman-sweep}"
