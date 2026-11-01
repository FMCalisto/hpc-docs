---
title: "12. Fine-tune a vision-language model on 4 GPUs"
parent: "Slurm"
nav_order: 12
---

# 12. Fine-tune a vision-language model on 4 GPUs (FSDP + LoRA)

[← Slurm](../../README.md)

| Nodes | CPUs | Memory | GPU | Time |
|---|---|---|---|---|
| 4 | 16 per node | 128 GB per node | 1 × L40S per node, 4 in all | about 2 h for 100 steps, mostly waiting on the network |

**First run:** Each node downloads the Qwen3.8-27B weights (55.6 GB, 1–2 min at 10 Gbit/s) and A-OKVQA (0.9 GB) into `/scratch/$USER/hf-cache`, in parallel: plan for 57 GB of your 100 GiB `/scratch` quota on each node, or 88 GB on a node that also has [example 9](../09-vlm-captions/README.md)'s FP8 weights. Loading the model takes 5–10 min per job. The LoRA adapter kept in `RESULTS_DIR` is about 100 MB. Needs the [examples environment](../../README.md#before-you-start), which includes `transformers`, `peft`, `accelerate` and `datasets`. See [Downloads and disk space](../../README.md#downloads-and-disk-space).

> ⚠️ **This job saturates the network between the nodes. Ask an admin before running it, and only in a slot they reserve.**
>
> The model doesn't fit one GPU, so FSDP splits its weights four ways and, for every training step, each GPU fetches the other three quarters from the other nodes, twice (forward and backward). That is **about 80 GB received per GPU per step**, and the four nodes do it at the same time. Our nodes are linked by Ethernet, not InfiniBand or NVLink: at 10 Gbit/s (about 1.25 GB/s), one step takes **at least a minute of pure transfer**, and the GPUs sit idle most of that time.
>
> While it runs, it uses most of each node's network bandwidth. Your home and `/storage/shared` are on MooseFS, which uses the same network, so **everyone on those nodes sees slow file access**. Other multi-node jobs slow down too. Run it in a reservation outside working hours, keep `MAX_STEPS` small, and cancel it if you don't need it.
>
> For real fine-tuning of a 27B model on this cluster, QLoRA (a 4-bit model with LoRA) on **one** GPU is usually faster and doesn't touch the network. This example exists to show how sharded training across nodes works, and what it costs here.

Teaches **Qwen3.8-27B** to answer commonsense questions about photos ("What is the man by the bags waiting for? Options: skateboarder, train, delivery, cab."), using [A-OKVQA](https://allenai.org/project/a-okvqa/home) from Hugging Face's [the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) collection: 16 539 questions about COCO images, free to download.

Two techniques make a 27B model trainable on 48 GB cards:

- **FSDP** (Fully Sharded Data Parallel) keeps a quarter of the weights on each GPU and gathers each layer from the others just before it is needed.
- **LoRA** freezes the model and trains small adapter matrices added to its attention layers, about 0.1% of the parameters. Gradients and optimizer state are therefore tiny, and only the frozen weights cross the network.

## Run it

    cd ~/slurm-examples/12-vlm-finetune-fsdp
    sbatch --reservation=<name> --partition=<partition> job.sbatch     # in the slot an admin gave you

    # A quick check of the whole pipeline: 10 steps
    sbatch --reservation=<name> --partition=<partition> --export=ALL,MAX_STEPS=10 job.sbatch

Without a reservation, the script asks for the `scavenge` partition, the only one with four GPUs, where it can be preempted at any moment. Files: [`job.sbatch`](job.sbatch), [`finetune.py`](finetune.py)

## What you should see

`vlm-fsdp-<jobid>.out` shows each node staging the model, the share of trainable parameters (about 0.1%), an evaluation loss on 200 held-out questions **before** training, the training loss every 5 steps, and the evaluation loss again **at the end**, which should be lower. Note the seconds per step: that is the network cost described above.

`~/results/vlm-fsdp/<jobid>/` gets the LoRA adapter (`adapter/`, about 100 MB) and the TensorBoard logs. Load the adapter on top of the base model with `peft`'s `PeftModel.from_pretrained`.

## How it works

- **Stage first, in parallel.** The first `srun` step downloads the weights and dataset on all four nodes at once, into each node's `/scratch`, so training never reads them over the network.
- **One process per node.** `torchrun` starts one process per node with its own `--node_rank`, all meeting at the job's first node. The Hugging Face `Trainer` sets up FSDP from `fsdp="full_shard auto_wrap"`.
- **Rank 0 loads, the others receive.** `cpu_ram_efficient_loading` makes only rank 0 read the 54 GB into memory (hence `--mem=128G`) and send each GPU its shard.
- **Activation checkpointing** recomputes activations during the backward pass instead of storing them, which keeps memory per GPU within the L40S's 48 GB.
- **No gradient accumulation.** With FSDP, every micro-batch gathers the weights again, so accumulating would multiply the network traffic. The script uses a batch of 4 per GPU instead.
- **Results.** Rank 0 writes to `/scratch` on the first node; when the job ends the adapter and logs are copied to `RESULTS_DIR`, and the working folder is removed on every node. The large sharded checkpoints are not copied.

## Try next

- Measure it: run with `MAX_STEPS=10` on 2 and on 4 nodes (`--nodes=2`) and compare the seconds per step. With Ethernet, more nodes don't make this faster.
- Your own data: replace the `load_dataset` line with your images and question–answer pairs in the same format, staged to `/scratch` first.
- Serve the fine-tuned model with vLLM, as in [example 9](../09-vlm-captions/README.md): vLLM can load LoRA adapters (`--enable-lora`); check that it supports them for this model first.
