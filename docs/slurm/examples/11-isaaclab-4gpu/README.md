---
title: "11. Robot learning on 4 GPUs"
parent: "Slurm"
nav_order: 11
---

# 11. Robot learning on 4 GPUs (Isaac Lab, multi-node)

[← Slurm](../../README.md)

| Nodes | CPUs | Memory | GPU | Time |
|---|---|---|---|---|
| 4 | 16 per node | 64 GB per node | 1 × L40S per node, 4 in all | about 1 h for 3000 iterations |

**First run:** Downloads nothing new if you've done [example 7](../07-isaaclab-rl/README.md)'s setup. Each node compiles shaders the first time Isaac Sim runs there (5–15 min), and uses under 1 GB of `/scratch`. Each run keeps 0.1–1 GB of checkpoints and logs. See [Downloads and disk space](../../README.md#downloads-and-disk-space).

> **Ask an admin before running this.** Every node has one GPU, so 4 GPUs means 4 nodes. The `shared` partition has only two (`hpc02`, `hpc03`), so the script runs in `scavenge`: idle group nodes and `hpc01`. There, a job is preempted, and this one killed, as soon as a node's owners need it. For a run that must finish, ask your group's admin for a reservation and submit into it with `--reservation=<name>` (and the partition they give you). See [Who's who](../../../overview/contacts.md).

Takes [example 7](../07-isaaclab-rl/README.md) from one GPU to four. The H1 humanoid learns to walk on **rough terrain** (slopes, steps, random ground), a harder task than example 7's flat ground. Each GPU simulates 4096 robots, so PPO learns from **16 384 robots at once**. After every update the four GPUs average their gradients over the network.

Reinforcement learning suits a cluster whose nodes are linked by Ethernet: most of the work is simulation, which stays on each GPU, and only the policy's small gradients cross the network.

## Run it

Needs the Isaac Lab environment from [example 7](../07-isaaclab-rl/README.md#set-up-once) (`setup.sbatch`). Your home is the same on every node, so the nodes share it.

    cd ~/slurm-examples/11-isaaclab-4gpu
    sbatch job.sbatch

    # Compare: the same task on one GPU, with example 7's script
    cd ../07-isaaclab-rl
    sbatch --export=ALL,TASK=Isaac-Velocity-Rough-H1,MAX_ITERATIONS=3000 train.sbatch

File: [`job.sbatch`](job.sbatch). Change the task, robots per GPU or length with `--export=ALL,TASK=...,NUM_ENVS=...,MAX_ITERATIONS=...`, or the number of GPUs with `--nodes=2`.

## What you should see

`isaaclab-4gpu-<jobid>.out` shows Isaac Sim starting on four nodes, then RSL-RL's iteration log from rank 0. Each line includes the steps collected per second, which should be close to 4 times example 7's for the same task, and the mean reward, which rises as the robot learns. Checkpoints and TensorBoard logs are copied to `~/results/isaaclab/logs/rsl_rl/<experiment>/<date>/`.

Watch the trained policy with example 7's `play.sbatch` and the same task:

    cd ../07-isaaclab-rl
    sbatch --export=ALL,TASK=Isaac-Velocity-Rough-H1 play.sbatch

## How it works

- **One process per GPU, one GPU per node.** `--nodes=4 --ntasks-per-node=1 --gres=gpu:1` gives one GPU on each of four nodes, and `srun` runs Isaac Lab's `train_multigpu` launcher once on each node.
- **`train_multigpu`** wraps `torchrun`. Each node passes its own `--node_rank` (Slurm's `SLURM_NODEID`), and they all meet at `MASTER_ADDR`, the job's first node. The shared `--run_timestamp` makes them agree on one run name.
- **NCCL over the private network.** `NCCL_SOCKET_IFNAME=ens3` and `GLOO_SOCKET_IFNAME=ens3` keep the traffic between nodes on the cluster network.
- **Data and results.** Each node runs in `/scratch/$USER/<jobid>`. Rank 0, on the first node, writes the logs and checkpoints; the script copies them to `RESULTS_DIR` every 30 minutes and at the end, and removes the working folder on every node.
- **Preemption.** In `scavenge`, losing one node stops the whole run. The last copied checkpoint survives in `RESULTS_DIR`.

## Try next

- Find where scaling stops: run with `--nodes=1`, `2` and `4`, and compare the steps per second.
- More robots per GPU: `NUM_ENVS=8192` uses more of each L40S's memory and makes the network share of each step smaller.
- The same pattern for your own PyTorch training is in [example 10](../10-ddp-multinode/README.md).
