#!/bin/bash
# Describe images with Qwen3.8-27B (FP8), on one L40S.
#   sbatch job.sbatch                                     # 200 images of COCO val2017 (free download)
#   sbatch --export=ALL,IMAGES=/path/to/images,LIMIT=0 job.sbatch   # all of your own images
#SBATCH --job-name=vlm-captions
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

SIF="${SIF:-$HOME/containers/vllm-0.32.0.sif}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
LIMIT="${LIMIT:-200}"
# Where the captions are kept: your home by default, or your group's NAS,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/captions
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/vlm-captions}"

# Caches on this node's /scratch, reused by later jobs: the model's weights (about 30 GB)
# and the sample images.
export HF_HOME="/scratch/$USER/hf-cache"
DATASETS="/scratch/$USER/datasets"
WORK="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$DATASETS" "$WORK" "$RESULTS_DIR"
finish() {
    [ -f "$WORK/captions.jsonl" ] && cp "$WORK/captions.jsonl" "$RESULTS_DIR/captions-$SLURM_JOB_ID.jsonl"
    rm -rf "$WORK"
    echo "Captions in $RESULTS_DIR/captions-$SLURM_JOB_ID.jsonl"
}
trap finish EXIT
trap 'exit 143' TERM

if [ -n "${IMAGES:-}" ]; then
    # Your images, from your home or the NAS: copy them to the local disk first.
    rsync -a "$IMAGES/" "$WORK/images/"
    IMAGE_DIR="$WORK/images"
else
    # COCO 2017 validation images (5000 photos, 0.8 GB, CC BY 4.0), downloaded once per node.
    IMAGE_DIR="$DATASETS/coco-val2017"
    if [ ! -f "$IMAGE_DIR/.complete" ]; then
        mkdir -p "$IMAGE_DIR"
        curl -fsSL -o "$WORK/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
        python3 -m zipfile -e "$WORK/val2017.zip" "$WORK"
        mv "$WORK"/val2017/* "$IMAGE_DIR/"
        touch "$IMAGE_DIR/.complete"
    fi
fi

echo "Node $(hostname), images from $IMAGE_DIR"
srun apptainer exec --nv --cleanenv --bind /scratch \
    --env HF_HOME="$HF_HOME" --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$SIF" \
    python3 "$SLURM_SUBMIT_DIR/caption.py" \
        --images "$IMAGE_DIR" --out "$WORK/captions.jsonl" --model "$MODEL" --limit "$LIMIT"
