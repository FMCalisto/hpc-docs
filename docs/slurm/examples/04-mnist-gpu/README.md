---
title: "4. Train a network on a GPU"
parent: "Slurm"
nav_order: 4
---

# 4. Train a network on a GPU (MNIST)

[← Slurm](../../README.md)

| CPUs | Memory | GPU | Time |
|---|---|---|---|
| 8 | 32 GB | 1 × L40S | a few minutes |

**First run:** Downloads MNIST (60 MB, seconds; no setup) to `/scratch/$USER/datasets` on each new node; about 0.1 GB of `/scratch` and 5 MB of results. See [Downloads and disk space](../../README.md#downloads-and-disk-space).

The same model and script as [example 3](../03-mnist-cpu/README.md), on one GPU with mixed precision. Compare the epoch times of the two.

## Run it

Needs the [examples environment](../../README.md#before-you-start).

    cd ~/slurm-examples/04-mnist-gpu
    sbatch job.sbatch

File: [`job.sbatch`](job.sbatch). It runs [`../03-mnist-cpu/mnist.py`](../03-mnist-cpu/mnist.py), so keep both folders.

## What you should see

In `mnist-gpu-<jobid>.out`, something like:

    Node hpc03, CUDA_VISIBLE_DEVICES=0
    NVIDIA L40S, 46068 MiB, 595.xx
    device=cuda torch_threads=... workers=8
    GPU: NVIDIA L40S
    epoch 1: loss 0.0533  test acc 98.60%  epoch time 3.1s

The first epoch is slower than the next ones while CUDA warms up. The model goes to `~/results/mnist/<jobid>/`, as in example 3.

## How it works

- `--gres=gpu:1` asks for one whole GPU. Each node has one L40S, so the job runs on a node whose GPU is free: your group's node, `hpc02` or `hpc03`, or an idle node through `scavenge`.
- Slurm sets `CUDA_VISIBLE_DEVICES` and only lets the job see its GPU. Outside a job, `nvidia-smi` fails (see [No GPU outside a job](../../../user-guide/04-slurm-basics.md#no-gpu-outside-a-job)).
- `--amp` uses bfloat16/float16 where it is safe, which the L40S runs much faster than float32.
- A GPU job still needs CPUs to feed it: MNIST is tiny, but for images or video the data loader is often the bottleneck. If `nvidia-smi` in the job shows low GPU use, add CPUs before anything else.

## Watch the GPU while it runs

    srun --jobid=<jobid> --overlap --pty nvidia-smi -l 2

## Try next

MNIST is too small to keep an L40S busy. Raise `--batch-size` to 4096 and see the epoch time drop, then move on to [example 6](../06-yolo-voc/README.md), which does.
