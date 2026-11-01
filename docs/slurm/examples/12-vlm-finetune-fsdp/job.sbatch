#!/bin/bash
# LoRA fine-tuning of Qwen3.8-27B on 4 GPUs (4 nodes x 1 L40S) with FSDP.
# WARNING: moves about 80 GB per GPU per training step over the nodes' Ethernet.
# Ask an admin first, and run it in a reserved slot (see README).
#   sbatch job.sbatch
#   sbatch --partition=<p> --reservation=<r> job.sbatch
#SBATCH --job-name=vlm-fsdp
#SBATCH --partition=scavenge
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
source "${ENV_DIR:-$HOME/envs/slurm-examples}/bin/activate"

# Where the adapter and logs are kept: your home by default, or your group's NAS,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/vlm-fsdp
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/vlm-fsdp}"
# Model weights (54 GB) and the dataset (0.9 GB) are cached on each node's /scratch.
export HF_HOME="/scratch/$USER/hf-cache"
export WORK="/scratch/$USER/$SLURM_JOB_ID"   # on each node; rank 0 (this node) writes the output
export MODEL="${MODEL:-Qwen/Qwen3.8-27B}" MAX_STEPS="${MAX_STEPS:-100}"

finish() {
    mkdir -p "$RESULTS_DIR/$SLURM_JOB_ID"
    # The adapter and TensorBoard logs; not the sharded checkpoints, which are large.
    rsync -a --exclude 'checkpoint-*' "$WORK/out/" "$RESULTS_DIR/$SLURM_JOB_ID/" 2>/dev/null || true
    srun --ntasks-per-node=1 --overlap rm -rf "$WORK" || true   # clean up on every node
    echo "Adapter and logs in $RESULTS_DIR/$SLURM_JOB_ID"
}
trap finish EXIT
trap 'exit 143' TERM

# 1. Stage the model and the dataset on every node, in parallel (about 55 GB per node,
#    1–2 min at 10 Gbit/s; skipped where they are already cached).
srun --ntasks-per-node=1 bash -c '
    mkdir -p "$WORK/out" "$HF_HOME"
    hf download "$MODEL" --quiet >/dev/null
    python -c "import datasets; datasets.load_dataset(\"HuggingFaceM4/the_cauldron\", \"aokvqa\", split=\"train\")" >/dev/null
    echo "$(hostname): model and data ready"'

# 2. Train: one process per node, rank 0 on this node.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME=ens3 GLOO_SOCKET_IFNAME=ens3   # the nodes' private network
export ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

echo "Nodes: $SLURM_JOB_NODELIST  master: $MASTER_ADDR:$MASTER_PORT  model: $MODEL"
srun bash -c 'exec torchrun \
    --nnodes="$SLURM_NNODES" --nproc_per_node=1 --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$SLURM_SUBMIT_DIR/finetune.py" --model "$MODEL" --max-steps "$MAX_STEPS" --out "$WORK/out"'
