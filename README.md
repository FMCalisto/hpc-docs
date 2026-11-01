# hpc-docs

User documentation for the ISR HPC cluster: the shared Slurm nodes hpc01–hpc03 and the group nodes `hpc-<group>`. The pages in `docs/` are published as the [ISR HPC Knowledge Base](docs/README.md).

## Start here

New users should read the [User Guide](docs/user-guide/README.md) in order, starting with [Getting an account](docs/user-guide/01-getting-an-account.md).

| Topic | Page |
|---|---|
| Account and password | [Getting an account](docs/user-guide/01-getting-an-account.md) |
| SSH login, keys, copying files | [SSH login](docs/user-guide/02-ssh-login.md) |
| Where to keep data, quotas | [Storage](docs/user-guide/03-storage.md) |
| Containers | [Apptainer](docs/user-guide/09-apptainer.md) |
| Running jobs | [Slurm basics](docs/user-guide/04-slurm-basics.md), [job script examples](docs/user-guide/05-slurm-job-examples.md) |
| Complete example jobs | [Slurm examples](docs/slurm/README.md): CPU, GPU, Isaac Lab, YOLO, vision-language models, multi-node |
| Tiers and priority | [Access and priority](docs/slurm/access-priority.md) |
| Your group's node | [Group nodes](docs/user-guide/06-group-nodes.md) |
| Problems | [FAQ and troubleshooting](docs/user-guide/07-faq.md) |
| How the cluster is laid out | [Overview](docs/overview/README.md) |

## Help

Ask your research group's admin first. The list of admins is in [Who's who](docs/overview/contacts.md).

## Related repositories

- [slurm-testing-cases](https://github.com/hpc-isr/slurm-testing-cases): example workloads.
- [hpc-automation](https://github.com/hpc-isr/hpc-automation): playbooks that configure the cluster.
