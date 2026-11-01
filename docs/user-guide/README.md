---
title: "User Guide"
nav_order: 2
has_children: true
has_toc: false
---

# User Guide

[← Home](../README.md)

Everything a researcher needs to log in and run work on the ISR HPC nodes. Read in order the first time.

1. [Getting an account](01-getting-an-account.md): invitation, assigned login, password recovery.
2. [SSH login](02-ssh-login.md): connecting, SSH keys, `~/.ssh/config`, copying files, limits outside Slurm.
3. [Storage](03-storage.md): your home, its quota, and `/scratch`.
4. [SLURM basics](04-slurm-basics.md): where jobs run, tiers and limits, GPUs and shards, preemption.
5. [SLURM job script examples](05-slurm-job-examples.md): copy-paste scripts for CPU, GPU, arrays and pipelines. For complete workloads, see the [Slurm example jobs](../slurm/README.md).
6. [Group nodes](06-group-nodes.md): using your group's dedicated machine.
7. [FAQ & troubleshooting](07-faq.md)
8. [Isaac Sim](08-isaac-sim.md): running Isaac Sim on the cluster.
9. [Containers with Apptainer](09-apptainer.md): converting a Docker image to Apptainer and running it in a job.

## Quick start

1. Log in: [SSH login](02-ssh-login.md).
2. Check your Slurm account and tier: `sacctmgr show assoc user=$USER format=account,qos%25,defaultqos`.
3. Run the [CPU smoke job](05-slurm-job-examples.md#1-short-cpu-smoke-job).
4. Follow it with `squeue --me` and `sacct`; see [SLURM basics](04-slurm-basics.md).

For a guided first test of CPU and GPU jobs, see the [Researcher test journey](00-researcher-test-journey.md).

Need help? See [Who's who](../overview/contacts.md).
