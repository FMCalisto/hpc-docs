#!/bin/bash
# One array task per sensor noise level. Submit with ./submit.sh.
#SBATCH --job-name=kalman-sweep
#SBATCH --array=0-19%10
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
source "${ENV_DIR:-$HOME/envs/slurm-examples}/bin/activate"
export OMP_NUM_THREADS=1   # numpy on small matrices: one thread is fastest

# Results end up here. Your home by default; your group's NAS works too,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/kalman-sweep
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/kalman-sweep}"

# Work on the node's local disk; remove it when the task ends, however it ends.
WORK="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$WORK" "$RESULTS_DIR"
trap 'rm -rf "$WORK"' EXIT

# Task 0 → sigma 0.25, task 19 → sigma 5.0
SIGMA=$(python -c "print(0.25 * ($SLURM_ARRAY_TASK_ID + 1))")

python "$SLURM_SUBMIT_DIR/kalman.py" \
    --sigma "$SIGMA" --runs 500 --seed "$SLURM_ARRAY_TASK_ID" \
    --out "$WORK/sigma-$SIGMA.json"

cp "$WORK/sigma-$SIGMA.json" "$RESULTS_DIR/"
