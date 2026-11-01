---
title: "10. Training across several nodes"
parent: "Slurm"
nav_order: 10
---

# 10. Training across several nodes (PyTorch DDP)

[← Slurm](../../README.md)

| Nodes | CPUs | Memory | GPU | Time |
|---|---|---|---|---|
| 2 | 16 per node | 64 GB per node | 1 × L40S per node | about 10 min |

**First run:** Downloads CIFAR-10 (170 MB, seconds, under a minute to unpack) on each node; 0.4 GB of `/scratch` per node and 50 MB of results. See [Downloads and disk space](../../README.md#downloads-and-disk-space).

> **Ask an admin before running this.** Multi-node jobs haven't been tested on this cluster yet, and a two-node job takes both GPUs of the `shared` partition. Your group's admin can confirm that your tier allows it and pick a quiet time. See [Who's who](../../../overview/contacts.md).

Every node of the cluster has a single GPU, so using more than one GPU for one training run means using more than one node. This example trains ResNet-18 on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) with PyTorch's **DistributedDataParallel** (DDP): one process per GPU, each training on a different part of every batch, with the gradients averaged over the network after every step.

## Run it

Needs the [examples environment](../../README.md#before-you-start).

    cd ~/slurm-examples/10-ddp-multinode
    sbatch job.sbatch               # 2 nodes
    sbatch --nodes=1 job.sbatch     # 1 node, to compare

Files: [`job.sbatch`](job.sbatch), [`train_ddp.py`](train_ddp.py)

## What you should see

`ddp-cifar-<jobid>.out` shows one line per process, then per-epoch throughput from rank 0, and a summary:

    rank 0/2 on hpc02, GPU NVIDIA L40S
    rank 1/2 on hpc03, GPU NVIDIA L40S
    epoch 1/10: loss 1.412  25.3s  1,976 images/s
    ...
    {"gpus": 2, "nodes": 2, "global_batch": 512, "test_accuracy": 0.91, "seconds_per_epoch": 9.8, ...}

The summary and the model are copied to `~/results/ddp-cifar/<jobid>/`. Run it with `--nodes=1` and `--nodes=2` and put the two `result.json` files side by side:

| GPUs | Seconds per epoch | Images per second | Speed-up |
|---|---|---|---|
| 1 | from your run | | 1.0× |
| 2 | from your run | | |

Perfect scaling would be 2×. Over the nodes' Ethernet network, a small model like ResNet-18 gets noticeably less, because every step waits for the gradients to cross the network. Bigger models with more computation per step scale better.

## How it works

- **One task per node.** `--nodes=2 --ntasks-per-node=1 --gres=gpu:1` gives one GPU on each of two nodes, and `srun` starts one copy of the command on each.
- **torchrun** starts the training process on each node. The first node of the job is the meeting point (`MASTER_ADDR`), and each node passes its own `--node_rank` (Slurm's `SLURM_NODEID`), so rank 0, which writes the results, always runs on the first node, where the batch script runs.
- **NCCL** carries the gradients between GPUs. `NCCL_SOCKET_IFNAME=ens3` makes it use the nodes' private network.
- **Data on each node.** `/scratch` is local to each node, so the first `srun` step makes sure every node has its own copy of CIFAR-10, downloading it once per node. Training then reads only from the local disk.
- **Results.** Rank 0 writes to `/scratch` on the first node; when the job ends, the batch script copies that to `RESULTS_DIR` and removes the working folder on every node.
- **Learning rate.** The global batch grows with the number of GPUs, so the script scales the learning rate with it.

## Try next

- Use the same pattern for your own model: wrap it in `DDP`, use a `DistributedSampler`, and only save from rank 0.
- Isaac Lab trains across nodes the same way, with its `--distributed` flag ([example 7](../07-isaaclab-rl/README.md)).
- If a job hangs at start-up, run it again with `export NCCL_DEBUG=INFO` added to the script and send the log to your admin.
