#!/bin/bash
# Step 1 (CPU): camera poses with COLMAP, from your video or a free sample scene.
#SBATCH --job-name=splat-prepare
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/common.sh"
trap 'rm -rf "$WORK"' EXIT
trap 'exit 143' TERM

if [ ! -f "$SIF" ]; then
    echo "Converting $NS_IMAGE to $SIF (once)"
    # Cache and build files in this job's /scratch folder (about 20 GB at the peak), removed at the end.
    export APPTAINER_CACHEDIR="$WORK/apptainer-cache" APPTAINER_TMPDIR="$WORK/apptainer-tmp"
    mkdir -p "$(dirname "$SIF")" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
    apptainer pull "$SIF" "$NS_IMAGE"
fi

if [ -f "$SCENE_RESULTS/data/transforms.json" ]; then
    echo "Already prepared: $SCENE_RESULTS/data"
    exit 0
fi

if [ -n "${VIDEO:-}" ]; then
    # Frame extraction and COLMAP write thousands of small files: do it on /scratch.
    echo "Processing $VIDEO with COLMAP on $SLURM_CPUS_PER_TASK CPUs"
    cp "$VIDEO" "$WORK/input.${VIDEO##*.}"
    ns ns-process-data video --data "$WORK/input.${VIDEO##*.}" --output-dir "$WORK/data" \
        --num-frames-target 300 --matching-method sequential --no-gpu
else
    # Nerfstudio's free sample scene, already processed.
    echo "No VIDEO given: downloading nerfstudio's sample scene '$SCENE_NAME'"
    ns ns-download-data nerfstudio --capture-name "$SCENE_NAME" --save-dir "$WORK/download"
    mv "$WORK/download/nerfstudio/$SCENE_NAME" "$WORK/data"
fi

# Keep the frames and poses (not COLMAP's scratch database) for the training step.
rsync -a --exclude 'colmap/database.db' "$WORK/data/" "$SCENE_RESULTS/data/"
echo "Prepared $(ls "$SCENE_RESULTS/data/images" | wc -l) images in $SCENE_RESULTS/data"
