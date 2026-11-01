---
title: "6. Train a YOLO object detector"
parent: "Slurm"
nav_order: 6
---

# 6. Train a YOLO object detector (Pascal VOC)

[← Slurm](../../README.md)

| CPUs | Memory | GPU | Time |
|---|---|---|---|
| 16 | 64 GB | 1 × L40S | several hours for 100 epochs |

**First run:** Downloads Pascal VOC (2.8 GB, under a minute, then 5–10 min to unpack and convert) on each new node. Needs 4 GB of `/scratch` for the dataset (7 GB while unpacking) and 0.1–0.5 GB in `RESULTS_DIR` per run. See [Downloads and disk space](../../README.md#downloads-and-disk-space).

Trains an [Ultralytics YOLO11](https://docs.ultralytics.com/) detector on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), 20 object classes and about 16 000 training images. It is a realistic long GPU job, so it is built to **survive preemption and time limits**: it resumes from its last checkpoint when it starts again.

## Run it

Needs the [examples environment](../../README.md#before-you-start) (it includes `ultralytics`).

    cd ~/slurm-examples/06-yolo-voc

    # Quick check first: 3 epochs on the 128-image COCO sample, a few minutes
    sbatch --time=00:20:00 --export=ALL,DATA=coco128.yaml,EPOCHS=3,RUN_NAME=yolo-test job.sbatch

    # Then the real run
    sbatch job.sbatch

Files: [`job.sbatch`](job.sbatch), [`train_yolo.py`](train_yolo.py)

The first run on a node downloads VOC to `/scratch/$USER/datasets` and converts it. Later runs on that node reuse it. The script also stores that path in your Ultralytics settings (`~/.config/Ultralytics/settings.json`), so other Ultralytics jobs use it too.

## What you should see

`yolo-voc-<jobid>.out` shows Ultralytics' per-epoch progress, and ends with the validation of the best model. After 100 epochs, YOLO11n reaches roughly 0.7–0.8 mAP50 on VOC. Results, plots and weights end up in `~/results/yolo-voc/train/`:

| File | What |
|---|---|
| `weights/best.pt`, `weights/last.pt` | Best and latest model |
| `results.csv`, `results.png` | Loss and mAP per epoch |
| `confusion_matrix.png`, `val_batch*_pred.jpg` | What the model gets right and wrong |

## How it survives preemption

Jobs in `scavenge`, and low-tier jobs on a group node, can be preempted: Slurm sends SIGTERM, waits 180 seconds, and puts the job back in the queue. This script handles that:

- `#SBATCH --requeue` lets Slurm put it back in the queue, and `--open-mode=append` keeps the log of every attempt in the same file.
- Training runs on `/scratch/$USER/runs/yolo-voc`, on the node's local disk. Ultralytics saves `last.pt` there after every epoch, and the script copies the run to `RESULTS_DIR` (default `~/results/yolo-voc`) every 30 minutes and when the job ends, including after SIGTERM.
- On start, the script copies a previous run back from `RESULTS_DIR` to `/scratch` and resumes from its `last.pt`. The `/scratch` path is the same on every node, so the checkpoint finds its files wherever the job lands.
- A `done` file marks the end, so a job requeued after finishing doesn't train again.
- `srun` starts Python as a job step, so it receives the signal.

The same restart works after a **time limit**: if the job is stopped by `--time`, submit it again and it continues where it was. To start over, delete `~/results/yolo-voc`.

## Try next

- A bigger model: `--export=ALL,MODEL=yolo11s.pt` (or `m`, `l`, `x`). Raise `--time` and memory to match.
- Your own dataset: write a [dataset YAML](https://docs.ultralytics.com/datasets/detect/) pointing to images on `/scratch` and pass it as `DATA=/path/to/data.yaml`. Stage the images to `/scratch` first, as in [Staging a dataset](../../../user-guide/03-storage.md#staging-a-dataset).
- Watch training live with TensorBoard: `tensorboard --logdir ~/results/yolo-voc` in a [Jupyter shard](../05-jupyter-shard/README.md) session.
