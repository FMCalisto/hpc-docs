#!/bin/bash
#SBATCH --job-name=yolo-voc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
source "${ENV_DIR:-$HOME/envs/slurm-examples}/bin/activate"

RUN_NAME="${RUN_NAME:-yolo-voc}"
# Where the run is kept: your home by default, or your group's NAS,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/yolo-voc
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/$RUN_NAME}"
# Training writes to the node's local disk. The path is the same on every node,
# so a run resumed on another node finds its files where its checkpoint expects them.
WORK="/scratch/$USER/runs/$RUN_NAME"
mkdir -p "$RESULTS_DIR" "$WORK"
cd "$WORK"   # Ultralytics downloads the starting weights (yolo11n.pt) here

if [ -e "$RESULTS_DIR/done" ]; then
    echo "Already finished: results in $RESULTS_DIR/train"
    exit 0
fi

# Resuming (after preemption, a time limit, or on another node): bring the run back to /scratch.
if [ -f "$RESULTS_DIR/train/weights/last.pt" ]; then
    rsync -a --update "$RESULTS_DIR/" "$WORK/"
fi

sync_out() { rsync -a "$WORK/" "$RESULTS_DIR/"; }
# Copy checkpoints out every 30 min too, in case the node itself fails.
( while sleep 1800; do sync_out; done ) &
SYNC_PID=$!
finish() {
    kill "$SYNC_PID" 2>/dev/null || true
    sync_out
    rm -rf "$WORK"
    echo "Run saved in $RESULTS_DIR"
}
trap finish EXIT
trap 'exit 143' TERM

echo "=== $(date -Is) start on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"

# srun passes Slurm's SIGTERM (preemption) to Python. Ultralytics saves
# last.pt after every epoch, so the requeued job loses at most one epoch.
srun python "$SLURM_SUBMIT_DIR/train_yolo.py" \
    --data "${DATA:-VOC.yaml}" --model "${MODEL:-yolo11n.pt}" \
    --epochs "${EPOCHS:-100}" --run-dir "$WORK" \
    --datasets "/scratch/$USER/datasets"

touch "$WORK/done"
echo "=== $(date -Is) finished"
