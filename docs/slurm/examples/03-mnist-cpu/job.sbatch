#!/bin/bash
#SBATCH --job-name=mnist-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
source "${ENV_DIR:-$HOME/envs/slurm-examples}/bin/activate"

# Where results are kept: your home by default, or your group's NAS,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/mnist
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/mnist}"
DATA="/scratch/$USER/datasets"        # dataset cache on this node, reused by later jobs
WORK="/scratch/$USER/$SLURM_JOB_ID"   # this job's working space on the node's local disk
mkdir -p "$DATA" "$WORK/out"

# However the job ends (success, error, time limit), copy the results out and clean up.
finish() {
    mkdir -p "$RESULTS_DIR/$SLURM_JOB_ID"
    cp -r "$WORK/out/." "$RESULTS_DIR/$SLURM_JOB_ID/"
    rm -rf "$WORK"
    echo "Results in $RESULTS_DIR/$SLURM_JOB_ID"
}
trap finish EXIT
trap 'exit 143' TERM

# Use the cores Slurm gave the job, and no more: half for the model's
# math threads, half for the data loader workers.
THREADS=$(( SLURM_CPUS_PER_TASK / 2 ))
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

echo "Node $(hostname), $SLURM_CPUS_PER_TASK CPUs"
python "$SLURM_SUBMIT_DIR/mnist.py" --device cpu --epochs 3 \
    --workers "$THREADS" --data "$DATA" --out "$WORK/out"
