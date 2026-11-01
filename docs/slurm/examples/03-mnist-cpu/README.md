---
title: "3. Train a network on CPUs"
parent: "Slurm"
nav_order: 3
---

# 3. Train a network on CPUs (MNIST)

[← Slurm](../../README.md)

| CPUs | Memory | GPU | Time |
|---|---|---|---|
| 16 | 16 GB | none | a few minutes |

**First run:** Downloads MNIST (60 MB, seconds; no setup) to `/scratch/$USER/datasets` on each new node; about 0.1 GB of `/scratch` and 5 MB of results. See [Downloads and disk space](../../README.md#downloads-and-disk-space).

Trains a small convolutional network on the MNIST handwritten digits with PyTorch, using only CPUs. It shows how to make a multi-threaded program use exactly the cores Slurm gave the job, and how to keep a dataset on the node's `/scratch`.

## Run it

Needs the [examples environment](../../README.md#before-you-start).

    cd ~/slurm-examples/03-mnist-cpu
    sbatch job.sbatch

Files: [`job.sbatch`](job.sbatch), [`mnist.py`](mnist.py)

## What you should see

In `mnist-cpu-<jobid>.out`, three epochs reaching about 99% test accuracy (times depend on the node and its load):

    device=cpu torch_threads=8 workers=8
    epoch 1: loss 0.0712  test acc 98.40%  epoch time 25.3s
    ...

and the trained weights in `~/results/mnist/<jobid>/`. Note the epoch time, to compare with [example 4](../04-mnist-gpu/README.md) on the GPU.

## How it works

- **Threads.** PyTorch, numpy and most math libraries start one thread per core they can see. The script sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from `SLURM_CPUS_PER_TASK`, so the job doesn't run more threads than it has cores, which slows it down. Here half the cores compute and half run the data loader.
- **Dataset on `/scratch`.** [MNIST](https://yann.lecun.com/exdb/mnist/) (free, 50 MB) is downloaded to `/scratch/$USER/datasets` the first time the job runs on a node, then reused by later jobs on that node. `/scratch` is a local NVMe disk on each node, the fastest place to read a dataset from (see [Staging a dataset](../../../user-guide/03-storage.md#staging-a-dataset)).
- **Results copied out.** The model is written to `/scratch/$USER/<jobid>/out` and copied to `RESULTS_DIR` (default `~/results/mnist`) when the job ends, even if it fails or hits its time limit. Then the working folder is removed.
- **CPU-only limit.** A CPU-only job can use at most 48 cores per node (36 on `hpc01`); larger requests are rejected.

## Try next

Run it with 4, 16 and 32 cores (`sbatch -c 4 job.sbatch`, and so on) and compare the epoch times. A small model stops getting faster well before 48 cores, which is worth knowing before asking for a whole node.
