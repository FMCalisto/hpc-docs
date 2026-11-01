#!/bin/bash
# Submit the sweep, then the collector, which waits for the whole array to succeed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

sweep=$(sbatch --parsable --export=ALL sweep.sbatch)
collect=$(sbatch --parsable --export=ALL --dependency="afterok:$sweep" collect.sbatch)

echo "Sweep array: $sweep   Collector: $collect (starts when the sweep succeeds)"
echo "Follow with: squeue --me"
