---
title: "5. Jupyter on a GPU slice"
parent: "Slurm"
nav_order: 5
---

# 5. Jupyter on a GPU slice (shard)

[← Slurm](../../README.md)

| CPUs | Memory | GPU | Time |
|---|---|---|---|
| 4 | 32 GB | 1 shard (a sixth of `hpc01`'s L40S) | up to 8 h, 2 h here |

**First run:** Downloads nothing beyond the [examples environment](../../README.md#before-you-start). See [Downloads and disk space](../../README.md#downloads-and-disk-space).

For trying things out, debugging a model, or a notebook, a whole GPU is usually more than you need. The `interactive` partition on `hpc01` splits its GPU into six **shards**: up to six users share it at once, so a shard often starts sooner than a whole GPU.

This job starts JupyterLab on a shard and prints the SSH tunnel to reach it from your laptop.

## Run it

Needs the [examples environment](../../README.md#before-you-start) (it includes JupyterLab).

    cd ~/slurm-examples/05-jupyter-shard
    sbatch job.sbatch
    cat jupyter-<jobid>.out      # once the job is running

File: [`job.sbatch`](job.sbatch)

## Connect

The log tells you what to do. In short, on your laptop:

    ssh -N -L 8888:localhost:<port> <username>@hpc01.isr.tecnico.ulisboa.pt

then open `http://127.0.0.1:8888/lab?token=<token>` with the token from the log. You need to be on IST's network or VPN to reach `hpc01`, as for any SSH login.

## Limits of a shard

- **One shard per user** at a time, with at most 6 CPUs and 64 GB of memory. A second one stays pending with `QOSMaxGRESPerUser`.
- **No GPU memory isolation.** All six users share the 48 GB, so keep your use under about 8 GB. In PyTorch you can enforce that in the first cell:

      import torch
      torch.cuda.set_per_process_memory_fraction(8 / 48)

- **Time-shared compute.** Your code gets the GPU in turns with the others, so don't time benchmarks on a shard.
- Up to 8 hours. For a long training run, submit a batch job with a whole GPU ([example 4](../04-mnist-gpu/README.md) or [6](../06-yolo-voc/README.md)).

## Data in a notebook

Notebooks and code live in your home. For data you read heavily, copy it to `/scratch/$USER/...` on `hpc01` from the notebook first and read it from there, as the batch examples do (see [Data and results](../../README.md#data-and-results)). Save anything you want to keep back to your home or your group's NAS: `/scratch` is cleaned after 30 days.

## Without a notebook

For a plain shell on a shard:

    srun --gres=shard:1 -c 4 --mem=32G -t 2:00:00 --pty bash

The resources stay reserved until you `exit`, even while you are idle. Please exit or `scancel` when you're done.
