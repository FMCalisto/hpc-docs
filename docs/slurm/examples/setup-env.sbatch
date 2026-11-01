#!/bin/bash
# Create the Python environment shared by examples 2–6 and 10.
# Run it once, from this folder:  sbatch setup-env.sbatch
#SBATCH --job-name=setup-env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/slurm-examples}"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# Keep pip's download cache off your home quota.
export PIP_CACHE_DIR="/scratch/$USER/pip-cache"

python -m pip install --upgrade pip
python -m pip install -r "$SLURM_SUBMIT_DIR/requirements.txt"

python - <<'PY'
import torch, torchvision, ultralytics
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| ultralytics", ultralytics.__version__)
print("CUDA build:", torch.version.cuda)
PY
echo "Environment ready: $ENV_DIR"
