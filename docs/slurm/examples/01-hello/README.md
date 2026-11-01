---
title: "1. Hello cluster"
parent: "Slurm"
nav_order: 1
---

# 1. Hello cluster

[← Slurm](../../README.md)

| CPUs | Memory | GPU | Time |
|---|---|---|---|
| 1 | 1 GB | none | 5 min |

The smallest useful job. It prints where it ran, which account, partition and QOS Slurm chose for you, and every `SLURM_*` variable the job can read. Run it first, to check that your account works.

## Run it

    cd ~/slurm-examples/01-hello
    sbatch job.sbatch
    squeue --me

Script: [`job.sbatch`](job.sbatch)

## What you should see

`hello-<jobid>.out` in this folder, starting with:

    Hello from hpc02 at 2026-11-01T10:00:00+00:00
    User:        <username>
    Account:     <group>
    Partition:   shared
    CPUs:        1 (visible to this job: 1)

`nproc` shows 1 even though the node has 56 cores: a job only sees the CPUs it asked for.

## Try next

- Ask for 4 CPUs: `sbatch -c 4 job.sbatch`, and see `nproc` change.
- Charge another group, if you are in more than one: `sbatch --account=<group> job.sbatch`.
- Run the same commands without a script: `srun -c 1 --mem=1G -t 5 hostname`.
