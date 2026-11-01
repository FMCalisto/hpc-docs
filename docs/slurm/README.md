---
title: "Slurm"
nav_order: 3
has_children: true
has_toc: false
---

# Slurm

[← Home](../README.md)

Every job on the cluster goes through [Slurm](https://slurm.schedmd.com/): you say what your job needs (CPUs, memory, GPUs, time) and Slurm starts it on a node where those are free. This section collects what you need to run jobs, and ten complete example jobs, from a one-line test to training across several nodes.

| To learn | Read |
|---|---|
| Partitions, limits, shards, preemption, monitoring | [SLURM basics](../user-guide/04-slurm-basics.md) |
| Your tier and how to get a higher one | [Access and priority](access-priority.md) |
| Short copy-paste commands | [SLURM job examples](../user-guide/05-slurm-job-examples.md) |
| Real workloads, end to end | The examples below |

## Example jobs

Each example is a folder with its job scripts and a page explaining what it does, how to run it and what to expect. They go up in size, and each one introduces something new.

| # | Example | What it shows | Resources |
|---|---|---|---|
| 1 | [Hello cluster](examples/01-hello/README.md) | A first job: where it ran, what it got | 1 CPU, 5 min |
| 2 | [Parameter sweep on CPUs](examples/02-kalman-sweep/README.md) | Job arrays, a collector job with `afterok` (Kalman filter Monte Carlo) | 20 × 1 CPU |
| 3 | [Train a network on CPUs](examples/03-mnist-cpu/README.md) | Multi-threading with the cores you asked for (MNIST) | 16 CPUs |
| 4 | [Train a network on a GPU](examples/04-mnist-gpu/README.md) | A whole GPU, mixed precision, CPU vs GPU timing | 1 GPU |
| 5 | [Jupyter on a GPU slice](examples/05-jupyter-shard/README.md) | Interactive work on a shard of `hpc01`'s GPU, SSH tunnel | 1 shard |
| 6 | [Train a YOLO object detector](examples/06-yolo-voc/README.md) | A long job that survives preemption and time limits (Pascal VOC) | 1 GPU, hours |
| 7 | [Robot learning with Isaac Lab](examples/07-isaaclab-rl/README.md) | RL for a humanoid on Isaac Sim (pip install), streamed to your laptop | 1 GPU |
| 8 | [3D scene from a video](examples/08-gaussian-splatting/README.md) | A CPU → GPU pipeline in Apptainer: COLMAP, then Gaussian Splatting | 16 CPUs, then 1 GPU |
| 9 | [Label images with a vision-language model](examples/09-vlm-captions/README.md) | Qwen3.8-27B on one GPU with vLLM, no cloud API | 1 GPU |
| 10 | [Training across several nodes](examples/10-ddp-multinode/README.md) | PyTorch DDP over two nodes with NCCL (ask an admin first) | 2 nodes × 1 GPU |

> These examples are written for this cluster but **haven't all been run on it yet**. Timings and outputs on the example pages are estimates. If one fails, tell your group's admin (see [Who's who](../overview/contacts.md)) with the job ID and the log.

## Before you start

Get the examples into your home on `hpc01`:

    git clone https://github.com/hpc-isr/hpc-docs.git ~/hpc-docs
    cp -r ~/hpc-docs/docs/slurm/examples ~/slurm-examples
    cd ~/slurm-examples

Submit each example **from its own folder** (`cd ~/slurm-examples/03-mnist-cpu`, then `sbatch job.sbatch`): the scripts find their Python files through `$SLURM_SUBMIT_DIR`, and their logs land there.

Examples 2–6 and 10 share one Python environment with PyTorch, Ultralytics and JupyterLab. Create it once, in a job:

    cd ~/slurm-examples
    sbatch setup-env.sbatch          # about 15 min; creates ~/envs/slurm-examples

Files: [`setup-env.sbatch`](examples/setup-env.sbatch), [`requirements.txt`](examples/requirements.txt). Examples 7, 8 and 9 bring their own software (Isaac Lab in a uv environment, nerfstudio and vLLM as Apptainer images) and set it up on first use.

You don't need `--partition` or `--account`: Slurm sends each job to your group's node, the `shared` nodes or an idle node. If you are in more than one group, add `--account=<group>`. Every script sets `--time` and `--mem`; keep them close to what the job needs.

## Data and results

Every example follows the same pattern, which you should copy for your own jobs:

| What | Where | Why |
|---|---|---|
| Code, environments, job scripts | your home | The same on every node |
| Datasets, model weights, download caches | `/scratch/$USER/datasets`, `/scratch/$USER/hf-cache`, ... | Downloaded once per node, reused by later jobs there. `/scratch` is the node's local NVMe disk, the fastest storage |
| The job's working files | `/scratch/$USER/<jobid>` | Fast, and doesn't load the shared filesystem. Removed when the job ends |
| Results to keep | `RESULTS_DIR`: `~/results/<example>` by default | Copied out when the job ends, even when it fails or runs out of time |

To keep results somewhere else, such as your group's NAS, set `RESULTS_DIR` when you submit:

    sbatch --export=ALL,RESULTS_DIR=/storage/anubis-nas/<group>/results/mnist job.sbatch

In a script, the pattern is:

    RESULTS_DIR="${RESULTS_DIR:-$HOME/results/my-job}"
    WORK="/scratch/$USER/$SLURM_JOB_ID"
    mkdir -p "$WORK/out"

    finish() {                      # runs however the job ends
        mkdir -p "$RESULTS_DIR/$SLURM_JOB_ID"
        cp -r "$WORK/out/." "$RESULTS_DIR/$SLURM_JOB_ID/"
        rm -rf "$WORK"
    }
    trap finish EXIT
    trap 'exit 143' TERM            # time limit or preemption: still run finish

    rsync -a "$HOME/data/my-dataset/" "$WORK/data/"       # stage the input
    srun python train.py --data "$WORK/data" --out "$WORK/out"

`/scratch` isn't shared: a file there is only visible on that node, and files untouched for 30 days are deleted. Long jobs that may restart on another node (examples 6 and 7) also copy their checkpoints out every 30 minutes. More in [Storage](../user-guide/03-storage.md#scratch).

## Datasets used

All are free and downloaded by the jobs themselves, on first use on each node.

| Dataset | Size | Used in | Source |
|---|---|---|---|
| MNIST handwritten digits | 50 MB | 3, 4 | [yann.lecun.com/exdb/mnist](https://yann.lecun.com/exdb/mnist/), via torchvision |
| Pascal VOC 2007 + 2012 | 3 GB | 6 | [host.robots.ox.ac.uk/pascal/VOC](http://host.robots.ox.ac.uk/pascal/VOC/), via Ultralytics |
| COCO128 (quick test) | 7 MB | 6 | [Ultralytics](https://docs.ultralytics.com/datasets/detect/coco128/) |
| Isaac Sim robot assets | | 7 | NVIDIA, downloaded by Isaac Sim |
| Nerfstudio "poster" scene | 100 MB | 8 | [nerfstudio sample data](https://docs.nerf.studio/quickstart/existing_dataset.html) |
| COCO 2017 validation images | 0.8 GB | 9 | [cocodataset.org](https://cocodataset.org/#download) (CC BY 4.0) |
| CIFAR-10 | 170 MB | 10 | [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html), via torchvision |

The model weights (YOLO11, Qwen3.8-27B-FP8) are downloaded the same way, from Ultralytics and Hugging Face.
