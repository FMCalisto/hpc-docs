---
title: "8. 3D scene from a video"
parent: "Slurm"
nav_order: 8
---

# 8. 3D scene from a video (COLMAP + Gaussian Splatting pipeline)

[← Slurm](../../README.md)

| Step | CPUs | Memory | GPU | Time |
|---|---|---|---|---|
| 1. Prepare: camera poses with COLMAP | 16 | 32 GB | none | 10–60 min for a video |
| 2. Train and render | 8 | 64 GB | 1 × L40S | about 30 min |

Turns a short phone video of a room, an object or a robot's workspace into a **3D Gaussian Splatting** model you can fly through, using [nerfstudio](https://docs.nerf.studio/) from its official container. It is a two-step **pipeline**: a CPU job finds the camera positions, and a GPU job, which Slurm starts only if the first succeeds, trains the model and renders a video. Splitting it this way means the GPU isn't held while COLMAP runs on CPUs.

## Run it

    cd ~/slurm-examples/08-gaussian-splatting

    # Nerfstudio's free sample scene (a poster on a wall)
    ./submit.sh

    # Your own video: walk slowly around the scene for 30–60 s, then
    VIDEO=$HOME/videos/lab.mp4 SCENE_NAME=lab ./submit.sh

Files: [`submit.sh`](submit.sh), [`prepare.sbatch`](prepare.sbatch), [`train.sbatch`](train.sbatch), [`common.sh`](common.sh)

The first run converts the nerfstudio image to `~/containers/nerfstudio-1.1.3.sif` (a few GB of your home quota). Each scene is kept in `~/results/splats/<scene>/` (set `RESULTS_DIR` for another place, such as your group's NAS). Preparation is skipped when the scene is already there, so you can rerun `submit.sh` safely.

## What you should see

In `logs/`, one log per step. When both are done:

| File | What |
|---|---|
| `~/results/splats/<scene>/data/` | Extracted frames and camera poses (`transforms.json`) |
| `~/results/splats/<scene>/runs/render-<jobid>.mp4` | A fly-through video of the scene |
| `~/results/splats/<scene>/runs/export/splat.ply` | The model; drag it into a web viewer such as [SuperSplat](https://superspl.at/editor) on your laptop |
| `~/results/splats/<scene>/runs/data/splatfacto/<date>/` | Checkpoints, config and TensorBoard logs |

## How it works

- **`--dependency=afterok`.** `submit.sh` submits both jobs at once. The second waits, with reason `Dependency`, until the first ends successfully. If preparation fails, training never starts: cancel it, fix the problem, and run `submit.sh` again.
- **COLMAP on CPUs.** `ns-process-data` extracts about 300 frames and runs COLMAP's feature matching and bundle adjustment with `--no-gpu`, using the job's 16 cores. Sequential matching suits video, where neighbouring frames overlap.
- **One container, both steps.** `common.sh` wraps every `ns-*` command in `apptainer exec`, adding `--nv` only in the GPU job.
- **Work on `/scratch`, keep results elsewhere.** Frame extraction and COLMAP write thousands of small files, and training reads every image thousands of times: both run in `/scratch/$USER/<jobid>` on the node. Each step copies what the next one needs to `RESULTS_DIR`, so the GPU step can run on any node and stage the scene from there.

## Tips for your own videos

Move slowly and steadily, keep the scene lit evenly, avoid moving people and reflective surfaces, and go around the object rather than turning on the spot. If COLMAP finds poses for only a few frames, the log says so; try again with a slower video, or `--matching-method exhaustive` in `prepare.sbatch` for short ones.
