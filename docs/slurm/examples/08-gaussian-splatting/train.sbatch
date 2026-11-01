#!/bin/bash
# Step 2 (GPU): train a 3D Gaussian Splatting model, then render a fly-through video.
#SBATCH --job-name=splat-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/common.sh"
finish() {
    [ -d "$WORK/runs" ] && rsync -a "$WORK/runs/" "$SCENE_RESULTS/runs/"
    rm -rf "$WORK"
    echo "Results in $SCENE_RESULTS/runs"
}
trap finish EXIT
trap 'exit 143' TERM

# Stage the prepared scene to the node's local disk; training reads every image many times.
rsync -a "$SCENE_RESULTS/data/" "$WORK/data/"
cd "$WORK"

ns ns-train splatfacto --data "$WORK/data" --output-dir "$WORK/runs" \
    --max-num-iterations "${ITERATIONS:-30000}" \
    --vis tensorboard   # logs to TensorBoard; no web viewer server to leave running

config=$(ls -t "$WORK"/runs/*/splatfacto/*/config.yml | head -1)
echo "Trained: $config"

# A smooth camera path through the training views, as an MP4.
ns ns-render interpolate --load-config "$config" \
    --output-path "$WORK/runs/render-$SLURM_JOB_ID.mp4" --frame-rate 30 --interpolation-steps 10

# The splat itself, as a .ply you can open in a web viewer such as SuperSplat.
ns ns-export gaussian-splat --load-config "$config" --output-dir "$WORK/runs/export"
