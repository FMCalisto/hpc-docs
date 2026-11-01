#!/bin/bash
# Train a policy with Isaac Lab + RSL-RL (PPO), headless.
#   sbatch train.sbatch                                   # H1 humanoid learns to walk
#   sbatch --export=ALL,TASK=Isaac-Humanoid train.sbatch  # another task
#   sbatch --export=ALL,STREAM=1 train.sbatch             # watch it train (WebRTC)
#SBATCH --job-name=isaaclab-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
TASK="${TASK:-Isaac-Velocity-Flat-H1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
STREAM="${STREAM:-0}"
# Where checkpoints and logs are kept: your home by default, or your group's NAS,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/isaaclab
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/isaaclab}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export OMNI_KIT_ACCEPT_EULA=YES

# Isaac Lab writes logs/rsl_rl/<experiment>/<date>/ under the current folder:
# make that the node's local disk, and copy it out when the job ends.
WORK="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$WORK" "$RESULTS_DIR"
cd "$WORK"
sync_out() { [ -d "$WORK/logs" ] && rsync -a "$WORK/logs/" "$RESULTS_DIR/logs/" || true; }
( while sleep 1800; do sync_out; done ) &   # checkpoints out every 30 min as well
SYNC_PID=$!
finish() {
    kill "$SYNC_PID" 2>/dev/null || true
    sync_out
    cd / && rm -rf "$WORK"
    echo "Logs and checkpoints in $RESULTS_DIR/logs"
}
trap finish EXIT
trap 'exit 143' TERM

extra=()
if [ "$STREAM" = 1 ]; then
    source "$SLURM_SUBMIT_DIR/stream-info.sh"
    extra=(--livestream 2)
    # Rendering the viewport costs GPU time: fewer environments keep training usable.
    NUM_ENVS="${STREAM_NUM_ENVS:-256}"
fi

echo "Task $TASK, $NUM_ENVS envs, $MAX_ITERATIONS iterations, on $(hostname)"
srun uv run --project "$ISAACLAB_DIR" --extra isaacsim \
    isaaclab train --rl_library rsl_rl --task "$TASK" \
    --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
    "${extra[@]}" physics=isaacsim_physx
