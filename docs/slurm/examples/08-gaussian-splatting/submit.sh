#!/bin/bash
# Submit both steps; training starts only if preparation succeeds.
#   ./submit.sh                      # nerfstudio's sample scene
#   VIDEO=~/videos/lab.mp4 SCENE_NAME=lab ./submit.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

prep=$(sbatch --parsable --export=ALL prepare.sbatch)
train=$(sbatch --parsable --export=ALL --dependency="afterok:$prep" train.sbatch)

echo "Prepare: $prep   Train + render: $train (waits for $prep)"
echo "Follow with: squeue --me"
