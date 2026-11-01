---
title: "Home"
nav_order: 1
---

# ISR HPC Knowledge Base

Documentation for users of the HPC cluster of the **Institute for Systems and Robotics (ISR)**.

## Using the cluster → [User Guide](user-guide/README.md)

| Task | Page |
|---|---|
| Get an account, reset my password | [Getting an account](user-guide/01-getting-an-account.md) |
| Log in with SSH, set up keys, copy files | [SSH login](user-guide/02-ssh-login.md) |
| Know where to keep my data | [Storage](user-guide/03-storage.md) |
| Run jobs on the shared cluster | [SLURM basics](user-guide/04-slurm-basics.md) · [Job script examples](user-guide/05-slurm-job-examples.md) |
| Start from a complete example job: CPU, GPU, Isaac Lab, YOLO, vision-language models, multi-node | [Slurm example jobs](slurm/README.md) |
| Use my group's dedicated node | [Group nodes](user-guide/06-group-nodes.md) |
| Run Isaac Sim | [Isaac Sim](user-guide/08-isaac-sim.md) |
| Run a container | [Containers with Apptainer](user-guide/09-apptainer.md) |
| See the queue and nodes in a browser | [Slurm-web](https://slurm.hpc.isr.tecnico.ulisboa.pt) (see [SLURM basics](user-guide/04-slurm-basics.md#check-monitor-and-cancel)) |
| Something isn't working | [FAQ & troubleshooting](user-guide/07-faq.md) |
| Learn how the cluster is laid out | [Overview](overview/README.md) |

## Running jobs → [Slurm](slurm/README.md)

Ten example jobs, from a one-line test to training across several nodes, with the scripts to run them: [Slurm](slurm/README.md).

## Conventions used in these pages

- Commands are run as your normal user unless they start with `sudo`, which needs admin rights.
- Placeholders look like `<username>` or `<group>`. Replace them, including the angle brackets.
