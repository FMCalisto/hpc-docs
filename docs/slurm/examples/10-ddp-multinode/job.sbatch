#!/bin/bash
# Data-parallel training on several nodes, one GPU each. Ask an admin before running it.
#   sbatch job.sbatch               # 2 nodes (hpc02 + hpc03 on the shared partition)
#   sbatch --nodes=1 job.sbatch     # the same on 1 GPU, for the scaling comparison
#SBATCH --job-name=ddp-cifar
#SBATCH --partition=shared
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
source "${ENV_DIR:-$HOME/envs/slurm-examples}/bin/activate"

# Where the result is kept: your home by default, or your group's NAS,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/ddp-cifar
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/ddp-cifar}"
DATA="/scratch/$USER/datasets/cifar10"   # a copy on each node's local disk
WORK="/scratch/$USER/$SLURM_JOB_ID"      # on each node; rank 0 (this node) writes the results
export DATA WORK EPOCHS="${EPOCHS:-10}"   # srun passes these to the tasks on every node

finish() {
    mkdir -p "$RESULTS_DIR/$SLURM_JOB_ID"
    cp -r "$WORK/out/." "$RESULTS_DIR/$SLURM_JOB_ID/" 2>/dev/null || true
    srun --ntasks-per-node=1 --overlap rm -rf "$WORK" || true   # clean up on every node
    echo "Results in $RESULTS_DIR/$SLURM_JOB_ID"
}
trap finish EXIT
trap 'exit 143' TERM

# 1. Stage the data on every node: one task per node downloads CIFAR-10 (170 MB,
#    free) if that node doesn't have it yet.
srun --ntasks-per-node=1 bash -c '
    mkdir -p "$WORK/out" "$DATA"
    if [ ! -f "$DATA/.complete" ]; then
        python -c "import os, torchvision; torchvision.datasets.CIFAR10(os.environ[\"DATA\"], download=True)"
        touch "$DATA/.complete"
    fi'

# 2. Train. The first node of the job hosts the rendezvous; every node starts
#    torchrun with its own node rank, so rank 0 runs here, on the first node.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME=ens3   # the nodes' private network interface
export OMP_NUM_THREADS=4

echo "Nodes: $SLURM_JOB_NODELIST  master: $MASTER_ADDR:$MASTER_PORT"
srun bash -c 'exec torchrun \
    --nnodes="$SLURM_NNODES" --nproc_per_node=1 --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$SLURM_SUBMIT_DIR/train_ddp.py" --epochs "$EPOCHS" --data "$DATA" --out "$WORK/out"'
