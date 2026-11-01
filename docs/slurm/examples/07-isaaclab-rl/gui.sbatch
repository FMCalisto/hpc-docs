#!/bin/bash
# The full Isaac Sim editor, streamed to your laptop, as a batch job.
# Uses the shared Apptainer image through the user guide's launch script:
#   cp ~/hpc-docs/docs/user-guide/examples/launch_isaac_sim.sh ~/ && chmod +x ~/launch_isaac_sim.sh
#   sbatch gui.sbatch      then read the server and ports in isaac-gui-<jobid>.out
#SBATCH --job-name=isaac-gui
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
LAUNCH="${LAUNCH_SCRIPT:-$HOME/launch_isaac_sim.sh}"
[ -x "$LAUNCH" ] || { echo "Copy launch_isaac_sim.sh to $LAUNCH first (see the header of this script)." >&2; exit 1; }

echo "Stop the session with:  scancel $SLURM_JOB_ID"
# The script prints the server address and your own ports (from your UID), then runs Isaac Sim.
exec "$LAUNCH"
