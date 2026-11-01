# shellcheck shell=bash
# Shared settings for prepare.sbatch and train.sbatch.
NS_IMAGE="${NS_IMAGE:-docker://ghcr.io/nerfstudio-project/nerfstudio:1.1.3}"
SIF="${SIF:-$HOME/containers/nerfstudio-1.1.3.sif}"
SCENE_NAME="${SCENE_NAME:-poster}"
# Where each scene is kept: your home by default, or your group's NAS,
# e.g. RESULTS_DIR=/storage/anubis-nas/<group>/splats
RESULTS_DIR="${RESULTS_DIR:-$HOME/results/splats}"
SCENE_RESULTS="$RESULTS_DIR/$SCENE_NAME"     # data/ (frames + poses) and runs/ (models, renders)

# Each job works on the node's local disk and removes it at the end.
WORK="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$WORK" "$SCENE_RESULTS"

# Run a nerfstudio command (ns-*) in the container, with the GPU when the job has one.
ns() {
    local nv=()
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nv=(--nv)
    apptainer exec "${nv[@]}" --bind /scratch "$SIF" "$@"
}
