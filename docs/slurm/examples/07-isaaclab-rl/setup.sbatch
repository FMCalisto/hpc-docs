#!/bin/bash
# Install Isaac Lab 3.0 with Isaac Sim 6.2 from pip, in a uv environment in your home.
# Run once:  sbatch setup.sbatch   (downloads about 20 GB; takes 20–40 min)
#SBATCH --job-name=isaaclab-setup
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
ISAACLAB_BRANCH="${ISAACLAB_BRANCH:-release/3.0.0}"

# uv: a fast pip replacement that also installs the Python version Isaac Lab needs (3.12).
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Download cache on this node's /scratch, not in your home quota.
export UV_CACHE_DIR="/scratch/$USER/uv-cache"

if [ ! -d "$ISAACLAB_DIR/.git" ]; then
    git clone --depth 1 --branch "$ISAACLAB_BRANCH" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
cd "$ISAACLAB_DIR"

# Creates $ISAACLAB_DIR/.venv with Isaac Lab, Isaac Sim (pip wheels) and CUDA PyTorch.
uv sync --extra isaacsim

uv run --extra isaacsim python -c "import isaaclab, isaacsim, torch; print('isaaclab OK, torch', torch.__version__)"
echo "Isaac Lab ready in $ISAACLAB_DIR"
