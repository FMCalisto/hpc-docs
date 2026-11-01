#!/bin/bash
# Isaac Lab RL on 4 GPUs: 4 nodes x 1 L40S, 4096 robots per GPU (16384 in all).
# Needs the Isaac Lab environment from example 7 (../07-isaaclab-rl/setup.sbatch).
# Ask an admin first: 4 GPUs means 4 nodes, more than the shared partition has.
#   sbatch job.sbatch                                    # in scavenge (preemptible)
#   sbatch --partition=<p> --reservation=<r> job.sbatch  # in a slot an admin reserved
#SBATCH --job-name=isaaclab-4gpu
#SBATCH --partition=scavenge
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
TASK="${TASK:-Isaac-Velocity-Rough-H1}"
NUM_ENVS="${NUM_ENVS:-4096}"            # per GPU
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
# Where checkpoints and logs are kept: your home by default, or your group's NAS,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/isaaclab
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/isaaclab}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export OMNI_KIT_ACCEPT_EULA=YES

# Every node works in its own /scratch folder. Rank 0, on this (the first) node,
# writes the logs and checkpoints; they are copied out when the job ends.
export WORK="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$WORK" "$RESULTS_DIR"
sync_out() { [ -d "$WORK/logs" ] && rsync -a "$WORK/logs/" "$RESULTS_DIR/logs/" || true; }
( while sleep 1800; do sync_out; done ) &   # checkpoints out every 30 min as well
SYNC_PID=$!
finish() {
    kill "$SYNC_PID" 2>/dev/null || true
    sync_out
    srun --ntasks-per-node=1 --overlap rm -rf "$WORK" || true   # clean up on every node
    echo "Logs and checkpoints in $RESULTS_DIR/logs"
}
trap finish EXIT
trap 'exit 143' TERM

# Rendezvous on the first node; one run name shared by all ranks.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
RUN_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
export MASTER_ADDR MASTER_PORT RUN_TIMESTAMP ISAACLAB_DIR TASK NUM_ENVS MAX_ITERATIONS
export NCCL_SOCKET_IFNAME=ens3 GLOO_SOCKET_IFNAME=ens3   # the nodes' private network

echo "Task $TASK: $SLURM_NNODES GPUs x $NUM_ENVS envs on $SLURM_JOB_NODELIST (master $MASTER_ADDR)"

# One launcher per node; each starts one training process on its GPU.
srun bash -c 'cd "$WORK" && exec uv run --project "$ISAACLAB_DIR" --extra isaacsim \
    isaaclab train_multigpu --rl_library rsl_rl \
    --nnodes "$SLURM_NNODES" --node_rank "$SLURM_NODEID" --num_gpus 1 \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    --run_timestamp "$RUN_TIMESTAMP" \
    --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
    physics=isaacsim_physx'
