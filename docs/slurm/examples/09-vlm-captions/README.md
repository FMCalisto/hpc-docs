---
title: "9. Label images with a vision-language model"
parent: "Slurm"
nav_order: 9
---

# 9. Label images with a vision-language model (Qwen3.8 + vLLM)

[← Slurm](../../README.md)

| CPUs | Memory | GPU | Time |
|---|---|---|---|
| 8 | 64 GB | 1 × L40S | a few minutes to load, then a few images per second |

Runs **Qwen3.8-27B**, a recent open-weight vision-language model, on one GPU of the cluster, with no cloud API and no data leaving ISR. The job reads a folder of images (COCO's sample photos, or your own robot camera frames, underwater footage or lab photos) and writes, for each one, a caption, the objects in it, the kind of environment and the hazards a mobile robot should avoid, as JSON lines.

It uses [vLLM](https://docs.vllm.ai/), the standard engine for serving language models fast, from its official container run with Apptainer.

**Why FP8.** Qwen3.8-27B in 16-bit precision needs about 54 GB, more than the L40S's 48 GB. The official FP8 version, [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), takes about 30 GB and leaves the rest for processing many images at once. The L40S runs FP8 natively.

## Run it

    cd ~/slurm-examples/09-vlm-captions

    # Once: convert the vLLM image to ~/containers/vllm-0.32.0.sif (about 10 GB)
    sbatch pull.sbatch

    # 200 photos from COCO val2017, a free dataset downloaded on first use
    sbatch job.sbatch

    # Or every image in a folder of your own (.jpg, .png, .webp)
    sbatch --export=ALL,IMAGES=$HOME/data/my-images,LIMIT=0 job.sbatch

Files: [`pull.sbatch`](pull.sbatch), [`job.sbatch`](job.sbatch), [`caption.py`](caption.py)

## Data and results

| What | Where | Why |
|---|---|---|
| Model weights (about 30 GB) | `/scratch/$USER/hf-cache` | Downloaded once per node from Hugging Face, reused by later jobs there |
| Sample images: [COCO val2017](https://cocodataset.org/#download) (5000 photos, 0.8 GB, CC BY 4.0) | `/scratch/$USER/datasets/coco-val2017` | Downloaded once per node |
| Your images (`IMAGES=...`) | copied to `/scratch/$USER/<jobid>/images` | Read from the local disk, not over the network |
| Captions | written to `/scratch/$USER/<jobid>/`, then copied to `~/results/vlm-captions/captions-<jobid>.jsonl` | Set `RESULTS_DIR` to keep them elsewhere, such as `/storage/anubis-nas/<group>/...` |

The first run on a node takes longer for the downloads. The weights count toward your [`/scratch` quota](../../../user-guide/03-storage.md#quota-1) on that node.

## What you should see

`vlm-captions-<jobid>.out` shows the model loading, then progress, and the results file has one line per image, for example:

    {"image": ".../000000000139.jpg", "caption": "A living room with a television, chairs and a dining table.",
     "objects": ["tv", "chair", "dining table", "potted plant"], "environment": "indoor", "hazards": ["chair legs"]}

## How it works

- **vLLM batches.** The script sends 32 images at a time and vLLM processes them together on the GPU, which is far faster than one by one. `--max-model-len 32768` caps each request's length (the model accepts 262k tokens), which leaves memory for bigger batches.
- **No thinking.** Qwen3.8 reasons step by step before answering by default. For labelling, the script turns that off (`enable_thinking: False`) and uses Qwen's recommended settings for direct answers. Remove that line for harder questions about an image, and expect slower answers.
- **The container brings its own CUDA.** The `cu129` build of vLLM runs on the nodes' NVIDIA driver; `--nv` gives it the GPU Slurm allocated. `--cleanenv` stops your session's Python settings from leaking into it.
- **Change the question** by editing `PROMPT` in `caption.py`: count objects, rate image quality, flag frames with people for anonymisation, or describe a robot's failure in a video frame.

## Try next

- **Serve it instead.** For interactive use, run vLLM's OpenAI-compatible server in a job, `apptainer exec --nv "$SIF" vllm serve Qwen/Qwen3.8-27B-FP8 --max-model-len 32768 --port <port>`, and reach it from your laptop through an SSH tunnel to that node, as in [example 5](../05-jupyter-shard/README.md). Remember the resources stay reserved while the server is up.
- **A smaller model** for faster labelling: any vision model vLLM supports works with `--export=ALL,MODEL=<huggingface-id>`.
