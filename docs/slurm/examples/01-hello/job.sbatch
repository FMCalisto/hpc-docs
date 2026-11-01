#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

echo "Hello from $(hostname) at $(date -Is)"
echo "User:        $(id -un)"
echo "Job ID:      $SLURM_JOB_ID"
echo "Account:     $SLURM_JOB_ACCOUNT"
echo "Partition:   $SLURM_JOB_PARTITION"
echo "QOS:         ${SLURM_JOB_QOS:-unknown}"
echo "CPUs:        $SLURM_CPUS_PER_TASK (visible to this job: $(nproc))"
echo "Memory:      ${SLURM_MEM_PER_NODE:-?} MB"
echo "Submit dir:  $SLURM_SUBMIT_DIR"
echo
echo "Your storage from this node:"
df -h "$HOME" /scratch 2>/dev/null
echo
echo "All Slurm variables for this job:"
env | grep '^SLURM_' | sort
