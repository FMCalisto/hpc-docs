#!/bin/bash
#SBATCH --job-name=mnist-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
source "${ENV_DIR:-$HOME/envs/slurm-examples}/bin/activate"

# Same layout as example 3: dataset cached on /scratch, work in /scratch, results copied out.
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/mnist}"   # or /storage/anubis-nas/<group>/mnist
DATA="/scratch/$USER/datasets"
WORK="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$DATA" "$WORK/out"

finish() {
    mkdir -p "$RESULTS_DIR/$SLURM_JOB_ID"
    cp -r "$WORK/out/." "$RESULTS_DIR/$SLURM_JOB_ID/"
    rm -rf "$WORK"
    echo "Results in $RESULTS_DIR/$SLURM_JOB_ID"
}
trap finish EXIT
trap 'exit 143' TERM

echo "Node $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Same script as example 3, on the GPU with mixed precision.
python "$SLURM_SUBMIT_DIR/../03-mnist-cpu/mnist.py" --device cuda --amp --epochs 3 \
    --batch-size 512 --workers "$SLURM_CPUS_PER_TASK" --data "$DATA" --out "$WORK/out"
