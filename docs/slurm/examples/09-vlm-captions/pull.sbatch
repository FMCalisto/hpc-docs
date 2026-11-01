#!/bin/bash
# Convert the vLLM Docker image to an Apptainer image, once:  sbatch pull.sbatch
#SBATCH --job-name=vllm-pull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# cu129 build: runs on the nodes' NVIDIA driver (CUDA 13.2).
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.32.0-cu129}"
SIF="${SIF:-$HOME/containers/vllm-0.32.0.sif}"

# Layer cache and build files on /scratch (about 35 GB at the peak), out of your
# home quota, and removed at the end: only the .sif (about 11 GB) is kept.
WORK="/scratch/$USER/$SLURM_JOB_ID"
export APPTAINER_CACHEDIR="$WORK/cache" APPTAINER_TMPDIR="$WORK/tmp"
mkdir -p "$(dirname "$SIF")" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
trap 'rm -rf "$WORK"' EXIT

apptainer pull "$SIF" "$VLLM_IMAGE"
ls -lh "$SIF"
