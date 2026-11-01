---
title: "SLURM job examples"
parent: "User Guide"
nav_order: 5
---

# SLURM job examples

[← User Guide](README.md) · [Home](../README.md)

Small starting examples. For complete workloads (training on CPUs and GPUs, Isaac Lab, YOLO, vision-language models, multi-node), see the [Slurm example jobs](../slurm/README.md). Your Slurm account is your research group (`dsor`, `irsg`, `laseeb`, `sipg` or `vislab`). Check it with `sacctmgr show assoc user=$USER format=account,qos%25,defaultqos`. Create the log directory before submitting:

    export SLURM_ACCOUNT="<your group>"
    mkdir -p "$HOME/slurm-logs"

Every user has the low tier, so `--qos=low` always works; drop it to use your own tier. There's no need for `--partition`: Slurm routes each job to your group's node, the shared nodes or an idle node (see [SLURM basics](04-slurm-basics.md#where-jobs-run)). Tiers are explained in [Access and priority](../slurm/access-priority.md).

## 1. Short CPU smoke job

    sbatch --account="$SLURM_ACCOUNT" --qos=low \
      --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
      --output="$HOME/slurm-logs/cpu-%j.out" \
      --error="$HOME/slurm-logs/cpu-%j.err" \
      --wrap='id -un; hostname; date -Is; echo "CPU smoke test passed."'

Expected: Slurm returns a job ID. After the job completes, inspect the output and accounting record using the commands in [SLURM basics](04-slurm-basics.md). A pending job is not a failure; do not resubmit it repeatedly.

## 2. CPU program with a user-owned environment

This uses a Python environment in your home, which every node sees (see [Storage](03-storage.md#software-environments)). Replace the paths with your own.

    cd "$HOME/slurm-logs"
    cat > "$HOME/slurm-logs/cpu-program.sbatch" <<'SBATCH'
    #!/bin/bash
    #SBATCH --job-name=cpu-program
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=2G
    #SBATCH --time=00:05:00
    #SBATCH --output=cpu-program-%j.out
    #SBATCH --error=cpu-program-%j.err

    source "$HOME/envs/my-project/bin/activate"
    python "$HOME/code/my-project/run.py" \
      --input "$HOME/data/input" \
      --output "$HOME/results/$SLURM_JOB_ID"
    SBATCH

    sbatch --account="$SLURM_ACCOUNT" --qos=low \
      "$HOME/slurm-logs/cpu-program.sbatch"

The example changes to the existing log directory before submission so the relative `#SBATCH` output paths resolve there. Ensure the output directory exists and the job can write to it.

## 3. One-GPU visibility smoke job

This checks that the job sees one NVIDIA L40S. It doesn't validate PyTorch, CUDA compatibility for a particular application, or multi-node communication.

    sbatch --account="$SLURM_ACCOUNT" --qos=low \
      --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --gres=gpu:1 \
      --time=00:05:00 \
      --output="$HOME/slurm-logs/gpu-%j.out" \
      --error="$HOME/slurm-logs/gpu-%j.err" \
      --wrap='hostname; printf "CUDA_VISIBLE_DEVICES=%s\n" "$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,name --format=csv,noheader'

Do not use this smoke test to benchmark performance or load the GPU. For a small slice of a GPU instead of a whole one, use `--gres=shard:1` (see [SLURM basics](04-slurm-basics.md#interactive-work)).

## 4. Small job array

This example creates two short tasks and asks Slurm to run at most one array element at a time. That array throttle does not prove that other per-user limits are enforced.

    sbatch --account="$SLURM_ACCOUNT" --qos=low \
      --array=0-1%1 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G \
      --time=00:02:00 \
      --output="$HOME/slurm-logs/array-%A_%a.out" \
      --wrap='echo "array task $SLURM_ARRAY_TASK_ID on $(hostname)"'

Expected: two task records and distinct output files. Do not scale the array without confirming current site limits and capacity.

## 5. Job dependency

The second job becomes eligible after the first succeeds. This does not guarantee when either job starts.

    producer=$(sbatch --parsable --account="$SLURM_ACCOUNT" --qos=low \
      --time=00:02:00 --output="$HOME/slurm-logs/producer-%j.out" \
      --wrap='echo producer; sleep 2')
    printf 'producer job: %s\n' "$producer"

    consumer=$(sbatch --parsable --account="$SLURM_ACCOUNT" --qos=low \
      --dependency="afterok:$producer" --time=00:02:00 \
      --output="$HOME/slurm-logs/consumer-%j.out" --wrap='echo consumer')
    printf 'consumer job: %s\n' "$consumer"

Inspect both jobs with squeue and sacct. Cancel only an exact job ID that you have verified belongs to you.

## 6. Interactive shell procedure

Use a short allocation and exit promptly. Interactive sessions can't be requeued, so if one is preempted it is killed:

    srun --account="$SLURM_ACCOUNT" --qos=low \
      --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
      --pty bash -l

Inside the allocated shell, run hostname, then type exit. If the prompt does not accept input, stop and ask the administrator; do not leave the allocation running or repeatedly resubmit.

## 7. Preemptible job that checkpoints

Jobs in `scavenge`, and low-tier jobs on a group node, can be preempted: they get SIGTERM, 180 seconds, and are then requeued. This script saves its state on SIGTERM and skips finished work when it starts again. Replace the program with your own; it must be able to resume from its checkpoint.

    cat > "$HOME/slurm-logs/resumable.sbatch" <<'SBATCH'
    #!/bin/bash
    #SBATCH --job-name=resumable
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=32G
    #SBATCH --time=1-00:00:00
    #SBATCH --requeue
    #SBATCH --open-mode=append
    #SBATCH --output=resumable-%j.out

    work="$HOME/runs/my-run"
    mkdir -p "$work"
    [ -e "$work/done" ] && { echo "already finished"; exit 0; }

    # srun forwards SIGTERM to the program, which should save a checkpoint
    # in $work and exit; Slurm then requeues the job.
    srun python "$HOME/code/my-project/train.py" --checkpoint-dir "$work" --resume
    touch "$work/done"
    SBATCH

    cd "$HOME/slurm-logs" && sbatch --account="$SLURM_ACCOUNT" resumable.sbatch

Check with `sacct -j <job-id>`: a preempted job shows `PREEMPTED` or `REQUEUED`, then runs again under the same job ID.

## Monitor and get help

    squeue -u "$USER" -o '%i|%a|%q|%T|%M|%R'
    sacct -X -j "<job-id>" --noheader \
      --format=JobIDRaw,User,Account,Partition,QOS,State,ExitCode,Elapsed,AllocTRES

For a problem report, include the job ID, the command or example used, approximate submit time, observed state or sanitized error, and what output you expected. Send this through the support route for your group from [Who's who](../overview/contacts.md). Do not include passwords, tokens, private keys, private research data, or unredacted logs in a public issue or message. See [FAQ and troubleshooting](07-faq.md).

## Not covered by these examples

A multi-node GPU/NCCL job has not been tested on this cluster; a script for one is in [example 10](../slurm/examples/10-ddp-multinode/README.md). Do not submit the multi-node case without specific SysAdmin coordination and confirmation of software, network, account, QoS, and test-window readiness.
