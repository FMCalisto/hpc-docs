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

# Layer cache and build files on /scratch, out of your home quota.
export APPTAINER_CACHEDIR="/scratch/$USER/apptainer-cache"
export APPTAINER_TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$(dirname "$SIF")" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

apptainer pull "$SIF" "$VLLM_IMAGE"
ls -lh "$SIF"
