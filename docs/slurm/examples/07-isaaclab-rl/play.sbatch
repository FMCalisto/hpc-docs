#!/bin/bash
# Stream the latest trained policy to the Isaac Sim WebRTC Streaming Client on your laptop.
#   sbatch play.sbatch           then read the address in isaaclab-play-<jobid>.out
#SBATCH --job-name=isaaclab-play
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
TASK="${TASK:-Isaac-Velocity-Flat-H1}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/isaaclab}"   # same as in train.sbatch

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export OMNI_KIT_ACCEPT_EULA=YES

# Stage the trained runs to the node's local disk and play from there.
WORK="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$WORK"
trap 'cd / && rm -rf "$WORK"' EXIT
trap 'exit 143' TERM
[ -d "$RESULTS_DIR/logs" ] || { echo "No trained runs in $RESULTS_DIR/logs: run train.sbatch first." >&2; exit 1; }
rsync -a "$RESULTS_DIR/logs" "$WORK/"
cd "$WORK"

source "$SLURM_SUBMIT_DIR/stream-info.sh"

# --checkpoint latest: the newest model_*.pt under logs/rsl_rl/ for this task.
srun uv run --project "$ISAACLAB_DIR" --extra isaacsim \
    isaaclab play --rl_library rsl_rl --task "$TASK" \
    --num_envs "${NUM_ENVS:-16}" --checkpoint latest --livestream 2 physics=isaacsim_physx
